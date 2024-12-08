import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets
from spikingjelly.activation_based import layer, functional, neuron, surrogate, encoding

tau = 2.0
T = 100
B = 64
lr = 1e-3
epochs = 100


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


def main():
    # init network
    net = SNN()
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        drop_last=True,     # 丢弃最后一个不完整的 batch
        num_workers=4,      # 多线程加载数据
        pin_memory=True,    # 利用数据在 CPU 和 GPU 之间传输的速度优化
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
    # init encoder
    encoder = encoding.PoissonEncoder()

    max_test_accuracy = -1.

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
            for t in range(T):
                # encode input
                encoded_image = encoder(image)
                out_fr += net(encoded_image)
            out_fr /= T
            # compute loss and backward
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
        print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, train_speed={train_speed:.2f} samples/s')

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
                out_fr = 0.
                for t in range(T):
                    encoded_image = encoder(image)
                    out_fr += net(encoded_image)
                out_fr /= T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_accuracy += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_accuracy /= test_samples
        print(f'Epoch {epoch + 1}: test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_speed={test_speed:.2f} samples/s')

        # save model
        save_max = False
        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            save_max = True
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_accuracy': max_test_accuracy,
        }
        if save_max:
            torch.save(checkpoint, './models/single_fully_connected_snn/best_model.pth')
        else:
            torch.save(checkpoint, './models/single_fully_connected_snn/latest_model.pth')

    # save final data
    net.eval()
    # register hooks
    output_layer = net.layer[-1]
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(module, input, output):
        module.v_seq.append(module.v.unsqueeze(0))
        module.s_seq.append(output.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)
    # evaluate
    with torch.no_grad():
        image, label = test_dataset[0]
        image = image.to(device)
        out_fr = 0.
        for t in range(T):
            encoded_image = encoder(image)
            out_fr += net(encoded_image)
        firing_rate = (out_fr / T).cpu().numpy()
        print(f'Firing rate: {firing_rate}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t = output_layer.v_seq.cpu().numpy().squeeze()
        s_t = output_layer.s_seq.cpu().numpy().squeeze()
        np.save("./models/single_fully_connected_snn/v_t.npy", v_t)
        np.save("./models/single_fully_connected_snn/s_t.npy", s_t)


if __name__ == '__main__':
    main()