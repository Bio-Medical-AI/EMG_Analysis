from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import pad
from functools import partial


class Conv2dCylindrical(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: tuple[int, int] = 0,
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device: Any = None,
                 dtype: Any = None) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                dilation=dilation,
                                groups=groups,
                                bias=bias,
                                device=device,
                                dtype=dtype)
        self.horizontal_pad = partial(pad, pad=(0, 0, padding[1], padding[1]), mode='circular')
        self.vertical_pad = partial(pad, pad=(padding[0], padding[0]), mode='constant', value=0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.horizontal_pad(x)
        v = self.vertical_pad(h)
        return self.conv2d(v)


class OriginalModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_width: int,
                 input_height: int,
                 channels: int):
        super().__init__()
        self.model = nn.Sequential(
            Conv2dCylindrical(channels, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Conv2dCylindrical(64, 64, (3, 3), (1, 1), (1, 1)),
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
