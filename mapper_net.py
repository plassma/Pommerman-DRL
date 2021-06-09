import torch
from torch import nn


class MappingNet(nn.Module):
    def __init__(self, num_stack, feature_space):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_stack, 256, kernel_size=7, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=5, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, feature_space)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).cuda()
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=0)
        return self.features(x)