import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes, image_size, batch_size, num_channels=3, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(batch_size, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        w, h = image_size // 4, image_size // 4
        self.fc = nn.Sequential(
            nn.Linear(batch_size * w * h, batch_size * w // 2 * h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(batch_size * w // 2 * h // 2, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x