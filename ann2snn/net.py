import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes, image_size, batch_size, num_channels=3, dropout=0.5):
        super().__init__()
        w, h = image_size // 4, image_size // 4
        self.layer = nn.Sequential(
            nn.Conv2d(num_channels, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(batch_size, batch_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(batch_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(batch_size * w * h, batch_size * w // 2 * h // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(batch_size * w // 2 * h // 2, num_classes),
        )

    def forward(self, x):
        return self.layer(x)