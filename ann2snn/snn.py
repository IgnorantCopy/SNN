from spikingjelly.activation_based import ann2snn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import sys
import os
import time
import datetime
from net import ConvNet
from tutorial.send_message import send_message
from data import FlowerDataset, Data


def config():
    parser = argparse.ArgumentParser(description='Convert ANN to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ann",                                        type=str,   help="path to ANN model", required=True)
    parser.add_argument("--dataset",            default="MNIST",        type=str,   help="dataset name", choices=["MNIST", "CIFAR10", "Flowers"])
    parser.add_argument("--dataset_root",       default="D:/DataSets",  type=str,   help="path to dataset")
    parser.add_argument("-m", "--mode",         default="max",          type=str,   help="convert mode", choices=["max", "99.9%", "1.0/2", "1.0/3", "1.0/4", "1.0/5"])
    parser.add_argument("-T", "--time_steps",   default=50,             type=int,   help="number of time steps to simulate")
    parser.add_argument("--gpu",                default=False,          type=bool,  help="use GPU")
    parser.add_argument("--log",                default=True,           type=bool,  help="save log file")
    parser.add_argument("--log_dir",            default="./logs",       type=str,   help="path to log directory")
    parser.add_argument("--model_dir",          default="./models",     type=str,   help="path to save model")
    parser.add_argument('--fine_tune',          default=False,          type=bool,  help='whether to fine tune the SNN model by STDP learning rule')
    return parser.parse_args()


def main():
    args = config()
    ann_path        = args.ann
    dataset_name    = args.dataset
    dataset_root    = args.dataset_root
    mode            = args.mode
    time_steps      = args.time_steps
    use_gpu         = args.gpu
    save_log        = args.log
    log_dir         = args.log_dir
    model_dir       = args.model_dir
    batch_size      = int(ann_path.split("_")[-3])
    fine_tune       = args.fine_tune

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = open(os.path.join(log_dir, f"log_snn{ann_path[ann_path.split('.')[1].rfind('ann')+3:]}_{time_steps}_{mode}"), 'w') if save_log else sys.stdout

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
        ann_net = ConvNet(num_of_labels, image_size, batch_size, num_channels=1)
    elif dataset_name == "CIFAR10":
        num_of_labels = 10
        image_size = 32
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = datasets.CIFAR10(root=dataset_root + "/CIFAR10", train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root=dataset_root + "/CIFAR10", train=False, transform=transform_test, download=True)
        ann_net = ConvNet(num_of_labels, image_size, batch_size, 3)
    elif dataset_name == "Flowers":
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

        ann_net = ConvNet(num_of_labels, image_size, batch_size, 3)
    else:
        log_file.write(f"Invalid dataset name: {dataset_name}\n")
        log_file.close()
        raise ValueError("Invalid dataset name")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    params = torch.load(ann_path)
    ann_net.load_state_dict(params["state_dict"])
    log_file.write(f"Loaded ANN model from {ann_path} with accuracy {params['accuracy']}\n")
    print(f"Loaded ANN model from {ann_path} with accuracy {params['accuracy']}")
    converter = ann2snn.Converter(mode=mode, dataloader=train_loader)
    snn_net = converter(ann_net)
    log_file.write(f"Converted ANN to SNN at {datetime.datetime.now()}\n")
    print(f"Converted ANN to SNN at {datetime.datetime.now()}")

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    snn_net.to(device)
    snn_net.eval()
    total = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            for m in snn_net.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            for t in range(time_steps):
                if t == 0:
                    out = snn_net(images)
                else:
                    out += snn_net(images)
            total += out.shape[0]
            correct += (torch.argmax(out, dim=1) == labels).sum().item()
        test_accuracy = correct / total
        log_file.write(f"Test accuracy: {test_accuracy:.4f}, Time elapsed: {time.time() - start_time:.2f} seconds\n")
        print(f"Test accuracy: {test_accuracy:.4f}, Time elapsed: {time.time() - start_time:.2f} seconds")
    torch.save({
        "state_dict": snn_net.state_dict(),
        "accuracy": test_accuracy,
    }, os.path.join(model_dir, f"snn{ann_path.split('.')[0][ann_path.rfind('ann')+3:]}_{time_steps}_{mode}.pth"))
    log_file.write(f"Saved SNN model at {datetime.datetime.now()}\n")
    print(f"Saved SNN model at {datetime.datetime.now()}")
    log_file.close()
    send_message()


if __name__ == '__main__':
    main()