from spikingjelly.activation_based import ann2snn, encoding
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import sys
import os
import time
import datetime
from net import ConvNet
from tutorial.send_message import send_message


def config():
    parser = argparse.ArgumentParser(description='Convert ANN to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ann",                                        type=str,   help="path to ANN model", required=True)
    parser.add_argument("--dataset",            default="MNIST",        type=str,   help="dataset name", choices=["MNIST"])
    parser.add_argument("--dataset_root",       default="E:/DataSets",  type=str,   help="path to dataset")
    parser.add_argument("-m", "--mode",         default="max",          type=str,   help="convert mode", choices=["max", "99.9%", "1.0/2", "1.0/3", "1.0/4", "1.0/5"])
    parser.add_argument("-T", "--time_steps",   default=100,            type=int,   help="number of time steps to simulate")
    parser.add_argument("--gpu",                default=False,          type=bool,  help="use GPU")
    parser.add_argument("--log",                default=True,           type=bool,  help="save log file")
    parser.add_argument("--log_dir",            default="./logs",       type=str,   help="path to log directory")
    parser.add_argument("--model_dir",          default="./models",     type=str,   help="path to save model")
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

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = open(os.path.join(log_dir, f"log_snn{ann_path[ann_path.rfind('ann')+3:-4]}_{time_steps}_{mode}"), 'w') if save_log else sys.stdout

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
    else:
        log_file.write(f"Invalid dataset name: {dataset_name}\n")
        log_file.close()
        raise ValueError("Invalid dataset name")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    params = torch.load(ann_path)
    ann_net.load_state_dict(params["state_dict"])
    log_file.write(f"Loaded ANN model from {ann_path} with accuracy {params['accuracy']}\n")
    converter = ann2snn.Converter(mode=mode, dataloader=train_loader)
    snn_net = converter(ann_net)
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
        log_file.write(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}, Time elapsed: {time.time() - start_time:.2f} seconds\n")
    torch.save({
        "state_dict": snn_net.state_dict(),
        "accuracy": test_accuracy,
    }, os.path.join(model_dir, f"snn{ann_path[ann_path.rfind('ann')+3:]}_{time_steps}_{mode}.pth"))
    log_file.write(f"Saved SNN model at {datetime.datetime.now()}\n")
    log_file.close()
    send_message()


if __name__ == '__main__':
    main()