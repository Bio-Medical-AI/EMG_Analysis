import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 transform: Compose = ToTensor(),
                 source_path_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'series'):
        self.records: pd.Series = data_frame[source_path_name]
        self.labels: pd.Series = data_frame[target_name]
        self.series: pd.Series = data_frame[series_name]
        self.transform = transform
        self.samples_amount = data_frame.shape[0]

    def __getitem__(self, index: int) -> dict:
        label = torch.tensor(self.labels.iloc[index]).long()
        series = torch.tensor(self.series.iloc[index]).long()
        out_index = torch.tensor(index).long()
        if self.transform is None:
            data = self.records.iloc[index]
        else:
            data = self.transform(self.records.iloc[index]).float()
        return {'data': data, 'label': label, 'series': series, 'index': out_index}

    def __len__(self) -> int:
        return self.samples_amount
