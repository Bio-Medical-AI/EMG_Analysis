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

from datasets.abstract_dataset import AbstractDataset


class AbstractDataModule(pl.LightningDataModule):
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
                 dataset: type or partial = AbstractDataset,
                 split_method: str = 'default'
                 ):
        super(AbstractDataModule, self).__init__()
        # path
        self.df_path: os.PathLike = df_path
        self.data: pd.DataFrame = pd.DataFrame()
        self.prepare_data()
        # data_parameters
        self.width = width
        self.height = height
        self.channels = channels
        self.num_classes = num_classes
        # split sizes
        self.k_folds = k_folds
        if self.k_folds < 3:
            self.train_vs_rest_size = train_vs_rest_size
            self.val_vs_test_size = val_vs_test_size
        else:
            self.k_folds = k_folds
            self.fold = 0
            self.folds = []
        # transforms
        self.train_transforms: Compose = train_transforms
        self.val_transforms: Compose = val_transforms
        self.test_transforms: Compose = test_transforms
        # column names
        self.source_name: str = source_name
        self.target_name: str = target_name
        self.series_name: str = series_name
        self.subject_name: str = subject_name
        # dataset parameters
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle_train: bool = shuffle_train
        # main dataframes
        self.splits: Dict[str, Index] = {}
        self.seed: int = randint(0, 2**32 - 1) if seed is None else seed
        # datasets
        self.dataset: type = dataset
        random.seed(self.seed)
        self.subjects = list(self.data[self.subject_name].unique())
        self.series = list(self.data[self.series_name].unique())
        self.targets = list(self.data[self.target_name].unique())
        self.split_method = split_method

    def prepare_data(self) -> None:
        self.data = pd.read_pickle(self.df_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.split_data()

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
        else:
            if not self.splits:
                tmp_data = self.data
                for f in range(self.k_folds):
                    series = self.get_random_series(tmp_data, proportion=1. / (self.k_folds - f))
                    self.folds.append(self.data.index[self.data[self.series_name].isin(series)])
                    tmp_data = tmp_data.loc[~tmp_data[self.series_name].isin(series)]

            self.splits['train'] = pd.Index([])
            for indexes in [
                fold for idx, fold in enumerate(self.folds) if idx not in [self.fold, (self.fold + 1) % self.k_folds]]:
                self.splits['train'] = self.splits['train'].union(indexes)
            self.splits['val'] = self.folds[self.fold]
            self.splits['test'] = self.folds[(self.fold + 1) % self.k_folds]

    def get_random_series(self, data: pd.DataFrame, proportion: float) -> list or None:
        if self.split_method == 'default':
            full_series = []
            for subject in self.subjects:
                for target in self.targets:
                    series = list(data.loc[data[self.subject_name] == subject].loc[data[self.target_name] == target]
                                  [self.series_name].unique())
                    full_series += random.sample(series, math.floor(len(series) * proportion))
            return full_series
        elif self.split_method == 'subject':
            subjects = random.sample(self.subjects, math.floor(len(self.subjects) * proportion))
            series = data.loc[data[self.subject_name].isin(subjects)][self.series_name].unique()
            return series
        return None

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['train']].reset_index(),
                         transform=self.train_transforms,
                         source_path_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name),
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
            self.dataset(data_frame=self.data.iloc[self.splits['val']].reset_index(),
                         transform=self.val_transforms,
                         source_path_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Prepares and returns test dataloader
        :return:
        """
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['test']].reset_index(),
                         transform=self.test_transforms,
                         source_path_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name),
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
