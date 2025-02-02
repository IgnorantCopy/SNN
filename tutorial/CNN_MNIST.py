import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class CNN(nn.Module):
    def __init__(self, batch_size=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, batch_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(batch_size, batch_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(batch_size * 7 * 7, batch_size * 4 * 4, bias=False),
            nn.ReLU(),

            nn.Linear(batch_size * 4 * 4, 10, bias=False),
        )

    def forward(self, x):
        return self.layer(x)


def main():
    # Hyperparameters
    batch_size = 64
    lr = 1e-3
    epochs = 100

    net = CNN(batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(net)

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='E:/DataSets',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataset = datasets.MNIST(
        root='E:/DataSets',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    max_test_accuracy = 0.
    start = time.time()
    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        train_loss = 0.
        for i, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}\n\tTrain Loss: {train_loss:.4f}\n\tTime: {time.time()-start_time:.2f}s")

        net.eval()
        start_time = time.time()
        test_loss = 0.
        accuracy = 0.
        with torch.no_grad():
            for i, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_dataloader)
        accuracy /= len(test_dataset)
        print(f"Epoch {epoch+1}/{epochs}\n\tTest Loss: {test_loss:.4f}\n\tAccuracy: {accuracy:.4f}\n\tTime: {time.time() - start_time:.2f}s")
        if accuracy > max_test_accuracy:
            max_test_accuracy = accuracy
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_accuracy': max_test_accuracy
            }, "./models/cnn_mnist.pth")
    print(f"Max Test Accuracy: {max_test_accuracy:.4f}")
    print(f"Total Time: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()