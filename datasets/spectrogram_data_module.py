import os
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from datasets import AbstractDataModule, SpectrogramDataset


class SpectrogramDataModule(AbstractDataModule):
    def __init__(self,
                 df_path: os.PathLike or str,
                 width: int,
                 height: int,
                 channels: int,
                 num_classes: int,
                 train_transforms: Compose = ToTensor(),
                 val_transforms: Compose = ToTensor(),
                 test_transforms: Compose = ToTensor(),
                 train_vs_rest_size: float = 0.8,
                 val_vs_test_size: float = 0.5,
                 source_name: str = 'path',
                 target_name: str = 'label',
                 series_name: str = 'spectrograms',
                 subject_name: str = 'subject',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None,
                 k_folds: int = 0,
                 dataset: type or partial = SpectrogramDataset,
                 split_method: str = 'default',
                 window_length: int = 1,
                 window_step: int = 1
                 ):
        super(SpectrogramDataModule, self).__init__(
            df_path,
            width,
            height,
            channels,
            num_classes,
            train_transforms,
            val_transforms,
            test_transforms,
            train_vs_rest_size,
            val_vs_test_size,
            source_name,
            target_name,
            series_name,
            subject_name,
            batch_size,
            num_workers,
            shuffle_train,
            seed,
            k_folds,
            dataset,
            split_method
        )
        self.window_length: int = window_length
        self.window_step: int = window_step
        self.locations: Dict[str, List] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        self.calculate_locations()

    def calculate_locations(self) -> None:
        if self.k_folds < 2:
            if not self.locations:
                self.locations = {}
                for key in self.splits.keys():
                    self.locations[key] = self.get_locations(self.data.iloc[self.splits[key]])
        else:
            if not self.locations:
                for idx, fold in enumerate(self.folds):
                    self.locations[str(idx)] = self.get_locations(self.data.iloc[fold])

            self.locations['train'] = []
            for indexes in \
                    [str(idx) for idx in range(self.k_folds) if idx not in [self.fold, (self.fold + 1) % self.k_folds]]:
                self.locations['train'] += self.locations[indexes]
            self.locations['val'] = self.locations[str(self.fold)]
            self.locations['test'] = self.locations[str((self.fold + 1) % self.k_folds)]

    def get_locations(self, data_frame: pd.DataFrame) -> list:
        locations = pd.Index([])
        series_ids = list(data_frame[self.series_name].unique())
        for s in series_ids:
            d = data_frame[data_frame[self.series_name] == s]
            d = d.iloc[:-self.window_length].iloc[::self.window_step, :].index
            locations = locations.union(d)
        return locations.tolist()

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data,
                         locations=self.locations['train'],
                         transform=self.train_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            shuffle=self.shuffle_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Prepares and returns validate dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data,
                         locations=self.locations['val'],
                         transform=self.val_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Prepares and returns test dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data,
                         locations=self.locations['test'],
                         transform=self.test_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
