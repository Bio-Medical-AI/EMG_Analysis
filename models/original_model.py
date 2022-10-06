import torch
from torch import Tensor
import torch.nn as nn


class OriginalModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_width: int,
                 input_height: int):
        super().__init__()
        self.model = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64 * input_width * input_height, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.weight_initialization()
        self.num_classes = num_classes

    def weight_initialization(self):
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
