import math
import os
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
                 df_path: os.PathLike | str,
                 train_transforms: Compose = ToTensor(),
                 val_transforms: Compose = ToTensor(),
                 test_transforms: Compose = ToTensor(),
                 train_vs_rest_size: float = 0.8,
                 val_vs_test_size: float = 0.5,
                 source_path_name: str = 'path',
                 target_name: str = 'label',
                 group_name: str = 'series',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None
                 ):
        super(AbstractDataModule, self).__init__()
        # path
        self.df_path: os.PathLike = df_path
        self.data: pd.DataFrame = pd.DataFrame()
        self.prepare_data()
        # split sizes
        self.train_vs_rest_size = train_vs_rest_size
        self.val_vs_test_size = val_vs_test_size
        # transforms
        self.train_transforms: Compose = train_transforms
        self.val_transforms: Compose = val_transforms
        self.test_transforms: Compose = test_transforms
        # column names
        self.source_path_name: str = source_path_name
        self.target_name: str = target_name
        self.group_name: str = group_name
        # dataset parameters
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle_train: bool = shuffle_train
        # main dataframes
        self.splits: Dict[str, Index] = {}
        self.seed: int = randint(0, 2**32 - 1) if seed is None else seed
        self.series = list(self.data[self.group_name].unique())
        random.seed(self.seed)

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.df_path)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or self.splits is {}:
            train_series = random.sample(self.series, math.floor(len(self.series) * self.train_vs_rest_size))
            val_series = random.sample(
                list(set(self.series) - set(train_series)),
                math.floor((len(self.series)-len(train_series)) * self.val_vs_test_size))
            test_series = list(set(self.series) - set(train_series) - set(val_series))
            self.splits['train'] = self.data.index[self.data[self.group_name].isin(train_series)]
            self.splits['val'] = self.data.index[self.data[self.group_name].isin(val_series)]
            self.splits['test'] = self.data.index[self.data[self.group_name].isin(test_series)]
            # self.splits['train'] = self.data\
            #     .sample(math.floor(self.data.shape[0] * self.train_vs_rest_size), random_state=self.seed).index
            # self.splits['val'] = self.data.iloc[~self.data.index.isin(self.splits['train'])]\
            #     .sample(math.floor((self.data.shape[0]-self.splits['train'].shape[0]) * self.val_vs_test_size),
            #             random_state=self.seed).index
            # self.splits['test'] = self.data.iloc[~self.data.index.isin(
            #     self.splits['train'].append(self.splits['val']))].index

    def train_dataloader(self) -> DataLoader:
        """
        Prepares and returns train dataloader
        :return:
        """
        return DataLoader(
            AbstractDataset(self.data.iloc[self.splits['train']],
                            self.train_transforms,
                            self.source_path_name,
                            self.target_name,
                            self.group_name),
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
            AbstractDataset(self.data.iloc[self.splits['val']],
                            self.val_transforms,
                            self.source_path_name,
                            self.target_name,
                            self.group_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Prepares and returns test dataloader
        :return:
        """
        return DataLoader(
            AbstractDataset(self.data.iloc[self.splits['test']],
                            self.test_transforms,
                            self.source_path_name,
                            self.target_name,
                            self.group_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
