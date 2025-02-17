import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from spikingjelly.activation_based import learning, layer, neuron, functional
import argparse
import sys
import os
import time
import datetime
from tutorial.send_message import send_message


def config():
    parser = argparse.ArgumentParser(description="Train SNN by hybrid training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",                default="MNIST",        type=str, help="dataset name", choices=["MNIST"])
    parser.add_argument("--dataset_root",           default="E:/DataSets",  type=str, help="path to dataset")
    parser.add_argument("-T", "--time_steps",       default=16,             type=int, help="number of time steps")
    parser.add_argument("--tau_pre",                default=2.,             type=float, help="time constant of presynaptic neuron")
    parser.add_argument("--tau_post",               default=100.,           type=float, help="time constant of postsynaptic neuron")
    parser.add_argument("--batch_size",             default=64,             type=int, help="batch size")
    parser.add_argument("-lr", "--learning_rate",   default=1e-3,           type=float, help="learning rate")
    parser.add_argument("--weight_decay",           default=5e-4,           type=float, help="weight decay")
    parser.add_argument("-e", "--epoches",          default=100,            type=int, help="number of epoches")
    parser.add_argument("--gpu",                    default=False,          type=bool,  help="use gpu")
    parser.add_argument("--log",                    default=True,           type=bool, help="save log file")
    parser.add_argument("--log_dir",                default="./logs",       type=str, help="path to log directory")
    parser.add_argument("--model_dir",              default="./models",     type=str, help="path to model directory")
    parser.add_argument("--pretrained_model",       default='',             type=str, help="path to pretrained model")
    return parser.parse_args()


def f_weight(x):
    return torch.clamp(x, -1, 1.)


def get_net(batch_size, input_channels, num_of_labels, image_size):
    w, h = image_size // 4, image_size // 4
    return nn.Sequential(
        layer.Conv2d(input_channels, batch_size, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),

        layer.Conv2d(batch_size, batch_size, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),

        layer.Flatten(),
        layer.Linear(batch_size * w * h, batch_size * w // 2 * h // 2, bias=False),
        neuron.IFNode(),

        layer.Linear(batch_size * w // 2 * h // 2, num_of_labels, bias=False),
        neuron.IFNode(),
    )


def main():
    args = config()
    dataset_name        = args.dataset
    dataset_root        = args.dataset_root
    time_steps          = args.time_steps
    tau_pre             = args.tau_pre
    tau_post            = args.tau_post
    batch_size          = args.batch_size
    lr                  = args.learning_rate
    epoches             = args.epoches
    use_gpu             = args.gpu
    save_log            = args.log
    log_dir             = args.log_dir
    model_dir           = args.model_dir
    pretrained_model    = args.pretrained_model

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file = open(os.path.join(log_dir, f"log_hybrid_{dataset_name}_{batch_size}_{lr:.0e}_{time_steps}"), 'w') if save_log else sys.stdout

    if dataset_name == "MNIST":
        image_size = 28
        num_of_labels = 10
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(root=dataset_root, train=False, download=True, transform=transform_test)
        net = get_net(batch_size, 1, num_of_labels, image_size)
    else:
        log_file.write(f"Invalid dataset name: {dataset_name}\n")
        log_file.close()
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    start_epoch = 0
    if pretrained_model:
        try:
            params = torch.load(pretrained_model)
            net.load_state_dict(params['state_dict'])
            start_epoch = params['epoch']
            log_file.write(f"Load pretrained model from {pretrained_model}\n")
        except Exception as e:
            log_file.write(f"Cannot load pretrained model from {pretrained_model}\n")
            log_file.write(str(e) + '\n')
            log_file.close()
            raise e

    functional.set_step_mode(net, 'm')
    instances_stdp = (layer.Conv2d,)
    stdp_learners = []
    for i in range(net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode='m', synapse=net[i], sn=net[i + 1], tau_pre=tau_pre, tau_post=tau_post,
                                     f_pre=f_weight, f_post=f_weight)
            )

    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gd = []
    for p in net.parameters():
        if p not in params_stdp_set:
            params_gd.append(p)

    optimizer_gd = Adam(params_gd, lr=lr, weight_decay=5e-4)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0., weight_decay=5e-4)
    gd_lr_scheduler = ReduceLROnPlateau(optimizer_gd, mode='min', patience=5)
    stdp_lr_scheduler = ReduceLROnPlateau(optimizer_stdp, mode='min', patience=5)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    net.to(device)


    best_acc = 0.
    log_file.write(f"Start training snn on {dataset_name} at {datetime.datetime.now()}\n")
    for epoch in range(start_epoch, epoches):
        start_time = time.time()
        net.train()
        train_loss = 0.
        train_acc = 0.
        train_samples = 0
        optimizer_gd.zero_grad()
        optimizer_stdp.zero_grad()
        for i, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)
            x_seq = x.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
            y = net(x_seq).mean(0)
            loss = F.cross_entropy(y, target)
            loss.backward()
            train_loss += loss.item() * target.numel()
            train_samples += target.numel()
            train_acc += (y.argmax(1) == target).sum().item()

            optimizer_stdp.zero_grad()
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad=True)

            optimizer_gd.step()
            optimizer_stdp.step()

            functional.reset_net(net)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()
        train_loss /= train_samples
        train_acc /= train_samples
        log_file.write(f"Epoch {epoch+1}:\n"
                       f"\ttrain loss: {train_loss:.4f}\n"
                       f"\ttrain acc: {train_acc:.4f}\n"
                       f"\ttime: {time.time() - start_time:.2f}s\n\n")
        gd_lr_scheduler.step(train_loss)
        stdp_lr_scheduler.step(train_loss)

        net.eval()
        test_loss = 0.
        test_acc = 0.
        test_samples = 0
        with torch.no_grad():
            for i, (x, target) in enumerate(test_loader):
                x, target = x.to(device), target.to(device)
                x_seq = x.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
                y = net(x_seq).mean(0)
                loss = F.cross_entropy(y, target)
                test_loss += loss.item() * target.numel()
                test_samples += target.numel()
                test_acc += (y.argmax(1) == target).sum().item()
        test_loss /= test_samples
        test_acc /= test_samples
        log_file.write(f"Epoch {epoch+1}:\n"
                       f"\ttest loss: {test_loss:.4f}\n"
                       f"\ttest acc: {test_acc:.4f}\n"
                       f"\ttime: {time.time() - start_time:.2f}s\n\n")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": net.state_dict(),
                "epoch": epoch+1,
                "accuracy": test_acc,
            }, os.path.join(model_dir, f"snn_{dataset_name}_{batch_size}_{lr:.0e}_{time_steps}.pth"))
            log_file.write(f"Save best model with test_acc: {best_acc:.4f}\n")
        log_file.write('-' * 50 + '\n')
        log_file.flush()
    log_file.write(f"End training snn on {dataset_name} at {datetime.datetime.now()} with best test_acc: {best_acc:.4f}\n")
    log_file.close()
    send_message()


if __name__ == '__main__':
    main()