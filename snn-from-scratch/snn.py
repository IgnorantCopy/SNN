import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional, encoding
import numpy as np
import sys
import os
import time
import datetime
from net import ConvSNN
# from tutorial.send_message import send_message


def config():
    parser = argparse.ArgumentParser(description="Train SNN from scratch", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",                default="MNIST",        type=str,   help="dataset name", choices=["MNIST", "CIFAR10", "Flowers"])
    parser.add_argument("--dataset_root",           default="E:/DataSets/",  type=str,   help="path to dataset")
    parser.add_argument("-T", "--time_steps",       default=100,            type=int,   help="number of time steps")
    parser.add_argument("-t", "--tau",              default=2.,             type=float, help="time constant of neuron")
    parser.add_argument("--batch_size",             default=64,             type=int,   help="batch size")
    parser.add_argument("-lr", "--learning_rate",   default=1e-3,           type=float, help="learning rate")
    parser.add_argument("--weight_decay",           default=5e-4,           type=float, help="weight decay")
    parser.add_argument("-e", "--epoches",          default=100,            type=int,   help="number of epoches")
    parser.add_argument("--optimizer",              default="Adam",         type=str,   help="optimizer", choices=["Adam", "SGD"])
    parser.add_argument("--cupy",                   default=True,          type=bool,  help="use cupy for GPU acceleration")
    parser.add_argument("--amp",                    default=False,          type=bool,  help="use automatic mixed precision")
    parser.add_argument("--log",                    default=True,           type=bool,  help="save log file")
    parser.add_argument("--log_dir",                default="./logs",       type=str,   help="path to log directory")
    parser.add_argument("--model_dir",              default="./models",     type=str,   help="path to model directory")
    parser.add_argument("--pretrained_model",       default='',             type=str,   help="path to pretrained model")
    return parser.parse_args()


def plot_loss(train_losses: list, test_losses: list):
    import plotly.graph_objects as go
    assert len(train_losses) == len(test_losses)
    x = np.arange(len(train_losses))
    fig = go.Figure(data=[
        go.Scatter(x=x, y=train_losses, name="train loss", mode="lines"),
        go.Scatter(x=x, y=test_losses, name="test loss", mode="lines"),
    ])
    fig.update_layout(title={
        "text": "Loss",
        "xanchor": "center",
        "yanchor": "top",
        "x": 0.5,
        "y": 0.9,
    }, xaxis_title="Epoch", yaxis_title="Loss")
    fig.show()


def main():
    args = config()
    dataset_name        = args.dataset
    dataset_root        = args.dataset_root
    time_steps          = args.time_steps
    tau                 = args.tau
    batch_size          = args.batch_size
    lr                  = args.learning_rate
    epoches             = args.epoches
    optimizer_name      = args.optimizer
    use_cupy            = args.cupy
    use_amp             = args.amp
    save_log            = args.log
    log_dir             = args.log_dir
    model_dir           = args.model_dir
    pretrained_model    = args.pretrained_model
    num_workers         = 0 if dataset_name in ["Flowers"] else 2

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file = open(os.path.join(log_dir, f"log_snn_{dataset_name}_{batch_size}_{optimizer_name}_{lr:.0e}_{time_steps}"), 'w') if save_log else sys.stdout

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
        model = ConvSNN(time_steps, tau, batch_size, 1, num_of_labels, image_size, use_cupy)
    elif dataset_name == "CIFAR10":
        num_of_labels = 10
        image_size = 32
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=True, transform=transform_train,
                                        download=True)
        test_dataset = datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=False, transform=transform_test,
                                        download=True)
        model = ConvSNN(time_steps, tau, batch_size, 3, num_of_labels, image_size, use_cupy)
    elif dataset_name == "Flowers":
        from data import FlowerDataset, Data

        num_of_labels = 16
        image_size = 32
        data = Data(dataset_name, dataset_root)
        train_data = data.sample(frac=0.8, random_state=2025)
        test_data = data.drop(train_data.index)

        train_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = FlowerDataset(train_data, train_transforms)
        test_dataset = FlowerDataset(test_data, test_transforms)
        model = ConvSNN(time_steps, tau, batch_size, 3, num_of_labels, image_size, use_cupy)
    else:
        log_file.write(f"Invalid dataset name: {dataset_name}\n")
        log_file.close()
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        log_file.write(f"Invalid optimizer name: {optimizer_name}\n")
        log_file.close()
        raise ValueError("Invalid optimizer name")
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cupy else "cpu")
    model.to(device)
    use_amp = use_amp and torch.cuda.is_available()
    scaler = amp.GradScaler() if use_amp else None
    encoder = encoding.PoissonEncoder(step_mode='m')

    start_epoch = 0
    best_acc = 0.
    if pretrained_model:
        try:
            params = torch.load(pretrained_model)
            model.load_state_dict(params['state_dict'])
            start_epoch = params['epoch']
            best_acc = params['accuracy']
            log_file.write(f"Load pretrained model from {pretrained_model}\n")
        except FileNotFoundError as e:
            log_file.write(f"Pretrained model not found: {pretrained_model}\n")
            log_file.write(str(e) + '\n')
            log_file.close()
            raise e


    log_file.write(f"Start training snn on {dataset_name} at {datetime.datetime.now()}\n")
    log_file.flush()
    tran_losses = []
    test_losses = []
    for epoch in range(start_epoch, epoches):
        start_time = time.time()
        train_loss = 0.
        train_acc = 0.
        train_samples = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels_onehot = F.one_hot(labels, num_classes=num_of_labels).float().to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with amp.autocast():
                    encoded_images = encoder(images)
                    out_fr = model(encoded_images)
                    loss = F.mse_loss(out_fr, labels_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                encoded_images = encoder(images)
                out_fr = model(encoded_images)
                loss = F.mse_loss(out_fr, labels_onehot)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * labels.numel()
            train_samples += labels.numel()
            train_acc += (out_fr.argmax(dim=1) == labels).sum().item()
            functional.reset_net(model)
        train_loss /= train_samples
        train_acc /= train_samples
        tran_losses.append(train_loss)
        log_file.write(f"Epoch {epoch+1}:\n"
                       f"\ttrain_loss: {train_loss:.4f}\n"
                       f"\ttrain_acc: {train_acc:.4f}\n"
                       f"\ttime: {time.time() - start_time:.2f}s\n\n")
        lr_scheduler.step(train_loss)

        model.eval()
        test_loss = 0.
        test_acc = 0.
        test_samples = 0
        start_time = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels_onehot = F.one_hot(labels, num_classes=num_of_labels).float().to(device)

                encoded_images = encoder(images)
                out_fr = model(encoded_images)
                loss = F.mse_loss(out_fr, labels_onehot)

                test_loss += loss.item() * labels.numel()
                test_samples += labels.numel()
                test_acc += (out_fr.argmax(dim=1) == labels).sum().item()
                functional.reset_net(model)
        test_loss /= test_samples
        test_acc /= test_samples
        test_losses.append(test_loss)
        log_file.write(f"\ttest_loss: {test_loss:.4f}\n"
                       f"\ttest_acc: {test_acc:.4f}\n"
                       f"\ttime: {time.time() - start_time:.2f}s\n")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch+1,
                "accuracy": test_acc,
            }, os.path.join(model_dir, f"snn_{dataset_name}_{batch_size}_{optimizer_name}_{lr:.0e}_{time_steps}.pth"))
            log_file.write(f"Save best model with test_acc: {best_acc:.4f}\n")
        log_file.write('-' * 50 + '\n')
        log_file.flush()
    log_file.write(f"End training snn on {dataset_name} at {datetime.datetime.now()} with best test_acc {best_acc:.4f}\n")
    log_file.close()
    # send_message()
    plot_loss(tran_losses, test_losses)


if __name__ == '__main__':
    main()