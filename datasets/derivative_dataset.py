from typing import List
import pandas as pd
import torch
from torchvision.transforms import Compose, ToTensor
from datasets.spectrogram_dataset import SpectrogramDataset


class DerivativeDataset(SpectrogramDataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 locations: List,
                 window_length: int,
                 transform: Compose = ToTensor(),
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms'):
        super().__init__(data_frame,
                         locations,
                         window_length,
                         transform,
                         source_name,
                         target_name,
                         series_name)

    def __getitem__(self, index: int) -> dict:
        output = super().__getitem__(index)
        output['data'] = add_derivative(output['data'])
        return output


def add_derivative(input: torch.Tensor) -> torch.Tensor:
    input = torch.squeeze(input)
    input = torch.reshape(input, (input.shape[0], -1))
    output = torch.cat([torch.zeros(1, input.shape[1]), input])[:-1]
    output = input - output
    output[0] = 0.
    output = torch.stack([input, output])
    return output
