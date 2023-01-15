import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


class UniLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 batch_first: bool,
                 dropout: float,
                 save_state: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            dropout=dropout)
        self.hidden = None
        self.save_state = save_state

    def forward(self, x: PackedSequence) -> PackedSequence:
        output, (h_0, c_0) = self.lstm(x, self.hidden) if self.hidden is not None else self.lstm(x)
        if self.save_state:
            self.hidden = (h_0.detach().clone(), c_0.detach().clone())
        return output

    def reset(self):
        self.hidden = None


class UnpackingSequencesToBatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: PackedSequence) -> Tensor:
        seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)
        return torch.cat([record[:length] for record, length in zip(seq_unpacked, lens_unpacked)])


class CRNN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super().__init__()
        self.model = nn.Sequential(
            UniLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                    dropout=dropout),
            UnpackingSequencesToBatch(),
            nn.Linear(hidden_size, num_classes))
        self.weight_initialization()
        self.num_classes = num_classes

    def weight_initialization(self):
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: PackedSequence) -> Tensor:
        return self.model(x)

    def reset_state(self):
        self.model[0].reset()
