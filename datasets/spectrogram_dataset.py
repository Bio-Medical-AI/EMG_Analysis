from typing import List
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from datasets.abstract_dataset import AbstractDataset


class SpectrogramDataset(AbstractDataset):
    """
    Base dataset for sequences of images.
    """
    def __init__(self,
                 data_frame: pd.DataFrame,
                 transform: Compose = ToTensor(),
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms',
                 window_length: int = 1):
        """
            Args:
                data_frame: DataFrame containing all data included in dataset
                transform: Transforms that are meant to be applied to single data sample
                source_name: Name of column in dataframe with data samples
                target_name: Name of column in dataframe with target classes
                series_name: Name of column in dataframe with series ID
                window_length: length of series of data
            """
        super().__init__(data_frame, transform, source_name, target_name, series_name)
        self.window_length = window_length

    def __getitem__(self, index: int) -> dict:
        label = int(self.labels.iloc[index])
        series = self.series.iloc[index]
        data = np.squeeze(
            np.dstack(self._get_window(index, self.window_length)))
        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=1)
        if self.transform is not None:
            data = self.transform(data).float()
        return {'data': data, 'label': label, 'spectrograms': series, 'index': index}

    def __len__(self) -> int:
        return self.samples_amount

    def _get_window(self, index: int, length: int) -> List[np.ndarray]:
        if index-length+1 < 0:
            return self._get_window(index, length-1) + [self.records.iloc[index]]
        if self.series.iloc[index] == self.series.iloc[index - length + 1]:
            return self.records.iloc[index-length+1:index+1].tolist()
        return self._get_window(index, length-1) + [self.records.iloc[index]]
