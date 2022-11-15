import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset
from datasets.abstract_dataset import AbstractDataset


class SpaceTimeDataset(AbstractDataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 locations: List,
                 window_length: int,
                 window_step: int,
                 transform: Compose = ToTensor(),
                 source_path_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'series'):
        super().__init__(data_frame, transform, source_path_name, target_name, series_name)

        self.locations = locations
        self.locations = pd.Index([])
        series_ids = list(data_frame[series_name].unique())
        for idx, s in enumerate(series_ids):
            d = data_frame[data_frame[series_name] == s]
            d = d.drop(d.tail(window_length).index, inplace=False).iloc[::window_step, :].index
            self.locations = self.locations.union(d)
        self.locations = self.locations.tolist()
        self.window_length = window_length
        self.samples_amount = len(self.locations)

    def __getitem__(self, index: int) -> dict:
        label = int(self.labels.iloc[self.locations[index]])
        series = self.series.iloc[self.locations[index]]
        data = np.squeeze(
            np.dstack(self.records[self.locations[index]:(self.locations[index] + self.window_length)].tolist()))
        if self.transform is not None:
            data = self.transform(data).float()
        return {'data': data, 'label': label, 'series': series, 'index': index}

    def __len__(self) -> int:
        return self.samples_amount
