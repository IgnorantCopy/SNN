import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer, surrogate


class ConvSNN(nn.Module):
    def __init__(self, time_steps, tau, batch_size, input_channels, num_of_labels, image_size, use_cupy=False):
        super().__init__()
        self.time_steps = time_steps
        w, h = image_size // 4, image_size // 4
        self.conv_fc = nn.Sequential(
            layer.Conv2d(input_channels, batch_size, kernel_size=3, padding=1),
            layer.BatchNorm2d(batch_size),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(batch_size, batch_size, kernel_size=3, padding=1),
            layer.BatchNorm2d(batch_size),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Linear(w * h * batch_size, w // 2 * h // 2 * batch_size),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

            layer.Linear(w // 2 * h // 2 * batch_size, num_of_labels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, backend='cupy') if use_cupy else None

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        return x_seq.mean(dim=0)