"""
Implementation of Lenet in pytorch.
Source: https://github.com/omegafragger/DDU/blob/main/net/lenet.py

Refernece:
[1] LeCun,  Y.,  Bottou,  L.,  Bengio,  Y.,  & Haffner,  P. (1998).
    Gradient-based  learning  applied  to  document  recognition.
    Proceedings of the IEEE, 86, 2278-2324.
"""
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes: int, input_channels: int, temp: float):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.temp = temp
        self.feature = None

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out) / self.temp


def LeNet_MNIST(num_classes):
    return LeNet(num_classes=num_classes, input_channels=1, temp=1.0)
