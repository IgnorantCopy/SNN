import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from spikingjelly.activation_based import ann2snn, encoding
import os
import sys
import datetime
import time
from net import ConvNet
from tutorial.send_message import send_message



def config():
    parser = argparse.ArgumentParser(description="Train ANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",                default="MNIST",        type=str,   help="dataset name", choices=["MNIST"])
    parser.add_argument("--dataset_root",           default="E:/DataSets/", type=str,   help="path to dataset")
    parser.add_argument("--batch_size",             default=64,             type=int,   help="batch size")
    parser.add_argument("-m", "--mode",             default="max",          type=str,   help="convert mode",
                        choices=["max", "99.9%", "1.0/2", "1.0/3", "1.0/4", "1.0/5"])
    parser.add_argument("-T", "--time_steps",       default=100,            type=int,   help="number of time steps to simulate")
    parser.add_argument("-lr", "--learning_rate",   default=1e-3,           type=float, help="learning rate")
    parser.add_argument("--weight_decay",           default=5e-4,           type=float, help="weight decay")
    parser.add_argument("-e", "--epoches",          default=100,            type=int,   help="number of epoches")
    parser.add_argument("--optimizer",              default="SGD",          type=str,   help="optimizer", choices=["SGD", "Adam"])
    parser.add_argument("--gpu",                    default=False,          type=bool,  help="use gpu")
    parser.add_argument("--log",                    default=True,           type=bool,  help="save log as a file")
    parser.add_argument("--log_dir",                default="./logs",       type=str,   help="path to save log")
    parser.add_argument("--model_dir",              default="./models",     type=str,   help="path to save model")
    parser.add_argument("--pretrained_model",       default='',             type=str,   help="path to pretrained model")
    return parser.parse_args()


def main():
    args = config()
    dataset_name        = args.dataset
    dataset_root        = args.dataset_root
    batch_size          = args.batch_size
    mode                = args.mode
    time_steps          = args.time_steps
    lr                  = args.learning_rate
    weight_decay        = args.weight_decay
    epoches             = args.epoches
    optimizer_name      = args.optimizer
    use_gpu             = args.gpu
    save_log            = args.log
    log_dir             = args.log_dir
    model_dir           = args.model_dir
    pretrained_model    = args.pretrained_model

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file = open(os.path.join(log_dir, f"log_ann_{dataset_name}_{batch_size}_{optimizer_name}_{lr:.0e}"), "w") if save_log else sys.stdout

    if dataset_name == "MNIST":
        num_of_labels = 10
        image_size = 28
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=dataset_root, train=True, transform=transform_train, download=True)
        test_dataset = datasets.MNIST(root=dataset_root, train=False, transform=transform_test, download=True)
        model = ConvNet(num_of_labels, image_size, batch_size, 1)
    else:
        log_file.write(f"Invalid dataset name: {dataset_name}\n")
        log_file.close()
        raise ValueError("Invalid dataset name")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        log_file.write(f"Invalid optimizer name: {optimizer_name}\n")
        log_file.close()
        raise ValueError("Invalid optimizer name")
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model.to(device)

    start_epoch = 0
    best_acc = 0.0
    if pretrained_model:
        try:
            params = torch.load(pretrained_model)
            model.load_state_dict(params['state_dict'])
            start_epoch = params['epoch']
            best_acc = params['accuracy']
            log_file.write("Load pretrained model from {}\n".format(pretrained_model))
        except FileNotFoundError as e:
            log_file.write(f"Cannot load pretrained model from {pretrained_model}\n")
            log_file.write(str(e) + "\n")
            log_file.close()
            raise e


    log_file.write(f"Start training ann on {dataset_name} at {datetime.datetime.now()}\n")
    log_file.flush()
    for epoch in range(start_epoch, epoches):
        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(labels.view_as(pred)).sum().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_dataset)
        lr_scheduler.step(train_loss)
        end_time = time.time()
        log_file.write(f"Epoch {epoch+1}/{epoches}:\n"
                       f"\ttrain loss: {train_loss:.4f}\n"
                       f"\ttrain acc: {train_acc:.4f}\n"
                       f"\ttime: {(end_time - start_time):.2f}s\n\n")

        start_time = time.time()
        test_loss = 0.0
        test_acc = 0.0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
                pred = outputs.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(test_loader)
        test_acc /= len(test_dataset)
        end_time = time.time()
        log_file.write(f"\ttest loss: {test_loss:.4f}\n"
                       f"\ttest acc: {test_acc:.4f}\n"
                       f"\ttime: {(end_time - start_time):.2f}s\n")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch+1,
                "accuracy": test_acc,
            }, os.path.join(model_dir, f"ann_{dataset_name}_{batch_size}_{optimizer_name}_{lr:.0e}.pth"))
            log_file.write(f"Save best model with test acc {best_acc:.4f}\n")
        log_file.write('-' * 50 + '\n')
        log_file.flush()
    log_file.write(f"End training ann on {dataset_name} at {datetime.datetime.now()} with best test accuracy {best_acc:.4f}\n")


    converter = ann2snn.Converter(mode=mode, dataloader=train_loader)
    snn_net = converter(model)
    log_file.write(f"Converted ANN to SNN at {datetime.datetime.now()}\n")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    snn_net.to(device)
    snn_net.eval()
    test_loss = 0.
    test_accuracy = 0.
    test_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels_onehot = F.one_hot(labels, num_of_labels).float()
            out_fr = 0.
            for t in range(time_steps):
                out_fr += snn_net(images)
            out_fr /= time_steps
            loss = F.mse_loss(out_fr, labels_onehot)
            test_loss += loss.item() * labels.numel()
            test_accuracy += (out_fr.argmax(dim=1) == labels).sum().item()
            test_samples += labels.numel()
        test_loss /= test_samples
        test_accuracy /= test_samples
        log_file.write(
            f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Time elapsed: {time.time() - start_time:.2f} seconds\n")
    torch.save({
        "state_dict": snn_net.state_dict(),
        "accuracy": test_accuracy,
    }, os.path.join(model_dir, f"snn_{dataset_name}_{batch_size}_{optimizer_name}_{lr:.0e}_{time_steps}_{mode}.pth"))
    log_file.write(f"Saved SNN model at {datetime.datetime.now()}\n")
    log_file.close()
    send_message()


if __name__ == '__main__':
    main()
