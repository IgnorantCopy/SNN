import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import amp
import torchvision
from spikingjelly.activation_based import neuron, functional, layer, surrogate, encoding
from spikingjelly import visualizing
import time
import datetime


# hyperparameters
T = 100
tau = 2.0
B = 64
lr = 1e-3
epochs = 100


"""
Network Architecture:
    {Conv2d -> BatchNorm2d -> IFNode -> MaxPool2d} ->
    {Conv2d -> BatchNorm2d -> IFNode -> MaxPool2d} ->
    {Linear -> IFNode}
"""
class CSNN(nn.Module):
    def __init__(self, T: int, C: int, use_cupy: bool = False):
        super().__init__()
        self.T = T

        self.layer = nn.Sequential(
            layer.Conv2d(1, C, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(C),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),    # output size: (C, H/2, W/2)

            layer.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(C),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),    # output size: (C, H/4, W/4)

            layer.Flatten(),
            layer.Linear(C * 7 * 7, C * 4 * 4, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

            layer.Linear(C * 4 * 4, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend="cupy") if use_cupy else None

    def forward(self, x: torch.Tensor):
        """
        :param x: shape = [B, C, H, W]
        :return:
        """
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)   # [T, B, C, H, W]
        x_seq = self.layer(x_seq)
        return x_seq.mean(dim=0)

    def spiking_encoder(self):
        """
        图片直接输入进网络，而不是经过编码后再输入。图片-脉冲 编码就是由网络的前三层完成的。
        :return: 网络前三层
        """
        return self.layer[0:3]


def main():
    # init network
    net = CSNN(T, B)
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = True if torch.cuda.is_available() else False      # amp(automatic mixed precision): 自动混合精度训练，可以节省显存并加快推理速度
    net.to(device)
    # init dataloader
    train_dataset = torchvision.datasets.MNIST(
        root='E:/DataSets',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='E:/DataSets',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=B,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=B,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    # init optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if not os.path.exists("./models/convolutional_snn"):
        os.makedirs("./models/convolutional_snn")
    writer = SummaryWriter("./models/convolutional_snn")
    scaler = None
    if use_amp:
        scaler = amp.GradScaler()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    max_test_accuracy = -1.
    train_accuracies = []
    test_accuracies = []
    start = time.time()
    for epoch in range(epochs):
        # train
        start_time = time.time()
        net.train()
        train_loss = 0
        train_accuracy = 0
        train_samples = 0
        for image, label in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, num_classes=10).float()

            out_fr = 0.
            if scaler is not None:
                with amp.autocast():
                    out_fr = net(image)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(image)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            # compute superparams
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_accuracy += (out_fr.argmax(1) == label).float().sum().item()
            # reset net
            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_accuracy /= train_samples
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, train_speed={train_speed:.2f} samples/s')

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)
        lr_scheduler.step()

        # test
        net.eval()
        test_loss = 0
        test_accuracy = 0
        test_samples = 0
        with torch.no_grad():
            for image, label in test_dataloader:
                image = image.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, num_classes=10).float()
                out_fr = net(image)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_accuracy += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_accuracy /= test_samples
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}: test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_speed={test_speed:.2f} samples/s')
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_accuracy", test_accuracy, epoch)

        # save model
        save_max = False
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            save_max = True
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_accuracy': max_test_accuracy,
        }
        if save_max:
            torch.save(checkpoint, './models/convolutional_snn/best_model.pth')
        torch.save(checkpoint, './models/convolutional_snn/latest_model.pth')

        print(f"Escape time: {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (epochs - epoch))).strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Training finished. Time: {time.time()-start:.2f} seconds.")
    # visualize
    x = np.arange(epochs)
    plt.plot(x, train_accuracies, label='train_accuracy')
    plt.plot(x, test_accuracies, label='test_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Training curves on FashionMNIST")
    plt.legend()
    plt.savefig("./models/convolutional_snn/training_curves.png", pad_inches=0.02)
    plt.show()


def visualize():
    # load model
    checkpoint = torch.load('./models/convolutional_snn/best_model.pth', map_location='cpu')
    net = CSNN(T, B)
    net.load_state_dict(checkpoint['net'])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    max_test_accuracy = checkpoint['max_test_accuracy']
    encoder = net.spiking_encoder()
    test_dataset = torchvision.datasets.FashionMNIST(
        root='E:/DataSets',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=B,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for image, label in test_dataloader:
            image = image.to(device)
            label = label.to(device)
            image_seq = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            spike_seq = encoder(image_seq)
            functional.reset_net(encoder)
            to_pil_image = torchvision.transforms.ToPILImage()

            if not os.path.exists("./models/convolutional_snn/visualize"):
                os.makedirs("./models/convolutional_snn/visualize")

            image = image.cpu()
            spike_seq = spike_seq.cpu()

            for i in range(label.shape[0]):
                os.mkdir(f"./models/convolutional_snn/visualize/{i}")
                to_pil_image(image[i]).save(f"./models/convolutional_snn/visualize/{i}/input.png")
                for t in range(T):
                    print(f"saving {i}-th sample with t={t}")
                    visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f"$S[{t}]$")
                    plt.savefig(f"./models/convolutional_snn/visualize/{i}/spike_{t}.png", pad_inches=0.02)
                    plt.clf()
            return


if __name__ == '__main__':
    main()