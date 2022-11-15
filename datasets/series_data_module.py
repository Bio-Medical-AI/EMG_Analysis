import math
import os
from functools import partial
from typing import Dict, Optional

import pandas as pd
import pytorch_lightning as pl
from pandas import Index
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from random import randint
import random
from torch.utils.data import Dataset

from datasets.abstract_dataset import AbstractDataset


class SeriesDataModule(pl.LightningDataModule):
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
                 series_name: str = 'series',
                 subject_name: str = 'subject',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None,
                 k_folds: int = 0,
                 dataset: type or partial = SpaceTimeDataset,
                 split_method: str = 'default',
                 window_length: int = 1,
                 window_step: int = 1
                 ):
        super(SeriesDataModule, self).__init__(
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
        self.window_length: int = window_length,
        self.window_step: int = window_step
        self.locations: Dict[str, List] = {}

    def split_data(self) -> None:
        if self.k_folds < 2:
            if not self.splits:
                train_series = self.get_random_series(self.data, proportion=self.train_vs_rest_size)
                non_train_data = self.data.loc[~self.data[self.series_name].isin(train_series)]
                val_series = self.get_random_series(non_train_data, proportion=self.val_vs_test_size)
                test_series = list(set(self.subjects) - set(train_series) - set(val_series))
                self.splits['train'] = self.data.index[self.data[self.subject_name].isin(train_series)]
                self.splits['val'] = self.data.index[self.data[self.subject_name].isin(val_series)]
                self.splits['test'] = self.data.index[self.data[self.subject_name].isin(test_series)]
                self.locations = {}
                for key in self.splits.keys():
                    self.locations[key] = self.get_locations(self.data.iloc[self.splits['test']].reset_index())
        else:
            if not self.splits:
                tmp_data = self.data
                for f in range(self.k_folds):
                    series = self.get_random_series(tmp_data, proportion=1. / (self.k_folds - f))
                    self.folds.append(self.data.index[self.data[self.series_name].isin(series)])
                    tmp_data = tmp_data.loc[~tmp_data[self.series_name].isin(series)]

                for idx, fold in enumerate(self.folds):
                    self.locations[str(idx)] = self.get_locations(self.data.iloc[self.splits['test']].reset_index())

            self.splits['train'] = pd.Index([])
            for indexes in [
                fold for idx, fold in enumerate(self.folds) if idx not in [self.fold, (self.fold + 1) % self.k_folds]]:
                self.splits['train'] = self.splits['train'].union(indexes)
            self.splits['val'] = self.folds[self.fold]
            self.splits['test'] = self.folds[(self.fold + 1) % self.k_folds]

            self.splits['train'] = []
            for indexes in \
                    [str(idx) for idx in range(self.k_folds) if idx not in [self.fold, (self.fold + 1) % self.k_folds]]:
                self.locations['train'].append(self.locations[indexes])
            self.locations['val'] = self.locations[str(self.fold)]
            self.locations['test'] = self.locations[str((self.fold + 1) % self.k_folds)]

    def get_locations(self, data_frame: pd.DataFrame) -> list:
        locations = pd.Index([])
        series_ids = list(data_frame[series_name].unique())
        for s in series_ids:
            d = data_frame[data_frame[series_name] == s]
            d = d.iloc[:-self.window_length].iloc[::self.window_step, :].index
            locations = locations.union(d)
        return locations.tolist()

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['train']],
                         locations=self.locations['train'],
                         transform=self.train_transforms,
                         source_path_name=self.source_name,
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
            self.dataset(data_frame=self.data.iloc[self.splits['val']],
                         locations=self.locations['val'],
                         transform=self.val_transforms,
                         source_path_name=self.source_name,
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
            self.dataset(data_frame=self.data.iloc[self.splits['test']],
                         locations=self.locations['test'],
                         transform=self.test_transforms,
                         source_path_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def get_data_parameters(self):
        return {'num_classes': self.num_classes, 'input_width': self.width, 'input_height': self.height,
                'channels': self.channels}

    def set_fold(self, fold: int):
        if self.k_folds > 2:
            self.fold = fold % self.k_folds

    def next_fold(self):
        if self.k_folds > 2:
            self.set_fold(self.fold + 1)
