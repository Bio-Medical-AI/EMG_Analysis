import torch
from torch import Tensor
import torch.nn as nn


class UniLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 batch_first: bool,
                 dropout: float,
                 sequence_length: int,
                 save_state: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
        self.hidden = None
        self.sequence_length = sequence_length
        self.save_state = save_state

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.size())
        length = shape[0]
        if length < self.sequence_length:
            shape[0] = self.sequence_length - length
            seq = torch.cat((x, torch.torch.zeros(shape, device=x.device)))
            output, (h_0, c_0) = self.lstm(seq, self.hidden) if self.hidden is not None else self.lstm(seq)
            output = output[:length]
        else:
            output, (h_0, c_0) = self.lstm(x, self.hidden) if self.hidden is not None else self.lstm(x)
        if self.save_state:
            self.hidden = (h_0.detach().clone(), c_0.detach().clone())
        return output

    def reset(self):
        self.hidden = None


class CRNN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_width: int,
                 input_height: int,
                 channels: int,
                 sequence_length: int):
        super().__init__()
        self.lstm = UniLSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, dropout=0.0,
                            sequence_length=sequence_length)
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, (3, 3), (1, 1), (1, 1)),
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
            self.lstm,
            nn.Linear(128, num_classes)
        )
        self.weight_initialization()
        self.num_classes = num_classes
        self.sequence_length = sequence_length

    def weight_initialization(self):
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def reset_state(self):
        self.lstm.reset()
