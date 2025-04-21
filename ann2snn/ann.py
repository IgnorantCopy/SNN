import argparse
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import os
import sys
import datetime
import time
from net import ConvNet
from tutorial.send_message import send_message
from PIL import Image


def config():
    parser = argparse.ArgumentParser(description="Train ANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset",                default="MNIST",        type=str,   help="dataset name", choices=["MNIST", "Flowers102", "CIFAR10", "Flowers"])
    parser.add_argument("--dataset_root",           default="E:/DataSets/", type=str,   help="path to dataset")
    parser.add_argument("--batch_size",             default=128,             type=int,   help="batch size")
    parser.add_argument("-lr", "--learning_rate",   default=1e-2,           type=float, help="learning rate")
    parser.add_argument("--weight_decay",           default=5e-4,           type=float, help="weight decay")
    parser.add_argument("-e", "--epoches",          default=100,            type=int,   help="number of epoches")
    parser.add_argument("--optimizer",              default="Adam",          type=str,   help="optimizer", choices=["SGD", "Adam"])
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
    elif dataset_name == "Flowers102":
        num_of_labels = 102
        image_size = 224
        transform_train = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.Flowers102(root=dataset_root, split="test", transform=transform_train, download=True)
        test_dataset = datasets.Flowers102(root=dataset_root, split="train", transform=transform_test, download=True)
        model = ConvNet(num_of_labels, image_size, batch_size, 3)
    elif dataset_name == "CIFAR10":
        num_of_labels = 10
        image_size = 32
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.CIFAR10(root=dataset_root+"CIFAR10", train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root=dataset_root+"CIFAR10", train=False, transform=transform_test, download=True)
        model = ConvNet(num_of_labels, image_size, batch_size, 3)
    elif dataset_name == "Flowers":
        num_of_labels = 16
        image_size = 224
        path = str(dataset_root)+"flowers"
        labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
                "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose", "sunflower",
                "tulip", "water_lily"]
        data = []
        for label in labels:
            img_dir = os.path.join(path, label)
            image_names = os.listdir(img_dir)
            for image_name in image_names:
                img_path = os.path.join(img_dir, image_name)
                data.append({"image_path": img_path, "label": label})
        data = pd.DataFrame(data)
        data["label"] = data["label"].map({label: i for i, label in enumerate(labels)})
        data.head()

        class FlowerDataset(Dataset):
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                img_path = self.data.iloc[index]["image_path"]
                label = self.data.iloc[index]["label"]
                image = Image.open(img_path)
                if self.transform:
                    image = self.transform(image)
                return image, label

        transforms_ = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = FlowerDataset(data, transform=transforms_)
        train_dataset, test_dataset = random_split(dataset, [13000, 2740])

        model = ConvNet(num_of_labels, image_size, batch_size, 3)
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
    log_file.close()
    send_message()


if __name__ == '__main__':
    main()
