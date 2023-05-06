import torch
import pandas as pd
from torchvision.transforms import Compose, ToTensor
from datasets.abstract_dataset import AbstractDataset


class SequenceDataset(AbstractDataset):
    """
    Base dataset for ordered images.
    """
    def __init__(self,
                 data_frame: pd.DataFrame,
                 transform: Compose = ToTensor(),
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms',
                 length_name: str = 'length'):
        """
        Args:
            data_frame: DataFrame containing all data included in dataset
            transform: Transforms that are meant to be applied to single data sample
            source_name: Name of column in dataframe with data samples
            target_name: Name of column in dataframe with target classes
            series_name: Name of column in dataframe with series ID
            length_name: Name of column in dataframe with length of the series
        """
        super().__init__(data_frame, transform, source_name, target_name, series_name)
        self.lengths: pd.Series = data_frame[length_name]

    def __getitem__(self, index: int) -> dict:
        out = super().__getitem__(index)
        out['length'] = torch.tensor(self.lengths.iloc[index]).long()
        return out

    def __len__(self) -> int:
        return self.samples_amount
