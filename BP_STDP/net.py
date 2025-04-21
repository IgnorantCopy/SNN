import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer


class SNN(nn.Module):
    def __init__(self, time_steps, batch_size, input_channels, num_of_labels, image_size, use_cupy=False):
        super().__init__()
        self.time_steps = time_steps
        w, h = image_size // 4, image_size // 4
        self.conv_fc = nn.Sequential(
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
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy') if use_cupy else None

    def forward(self, x_seq: torch.Tensor):
        x_seq = self.conv_fc(x_seq)
        return x_seq.mean(dim=0)