from typing import List
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from datasets.abstract_dataset import AbstractDataset


class SpectrogramDataset(AbstractDataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 locations: List,
                 window_length: int,
                 transform: Compose = ToTensor(),
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms'):
        super().__init__(data_frame, transform, source_name, target_name, series_name)

        self.locations = locations
        self.window_length = window_length
        self.samples_amount = len(self.locations)

    def __getitem__(self, index: int) -> dict:
        label = int(self.labels.iloc[self.locations[index]])
        series = self.series.iloc[self.locations[index]]
        data = np.squeeze(
            np.dstack(self.records[self.locations[index]:(self.locations[index] + self.window_length)].tolist()))
        if self.transform is not None:
            data = self.transform(data).float()
        return {'data': data, 'label': label, 'spectrograms': series, 'index': index}

    def __len__(self) -> int:
        return self.samples_amount
