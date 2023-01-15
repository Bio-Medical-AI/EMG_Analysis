import numpy as np
import torch
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from datasets import AbstractDataset


class AugmentedDataset(AbstractDataset):
    def __init__(self, data_frame: pd.DataFrame,
                 transform: Compose = ToTensor(),
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms'):
        super().__init__(data_frame, transform, source_name, target_name, series_name)
        self.rotations = self.records.iloc[0].shape[-2]
        self.series_step = min([self.series.max(), self.rotations])
        self.base_samples_amount = self.samples_amount
        self.samples_amount = self.samples_amount * self.rotations

    def __getitem__(self, index: int) -> dict:
        rotation = index // self.base_samples_amount
        base_idx = index % self.base_samples_amount
        label = torch.tensor(self.labels.iloc[base_idx]).long()
        series = torch.tensor(self.series.iloc[base_idx]).long() + rotation * self.series_step
        out_index = torch.tensor(index).long()

        data = np.roll(self.records.iloc[base_idx], rotation, axis=0)
        if self.transform is not None:
            data = self.transform(data).float()
        return {'data': data, 'label': label, 'spectrograms': series, 'index': out_index}

    def __len__(self) -> int:
        return self.samples_amount
