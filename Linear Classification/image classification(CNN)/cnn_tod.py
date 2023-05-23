import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_channels=3, classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5,5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5))
        self.fc1 = nn.Linear(140450, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
