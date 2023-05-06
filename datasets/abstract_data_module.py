import math
import os
from functools import partial
from typing import Dict, Optional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pandas import Index
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randint
import random

from datasets.spectrogram_dataset import SpectrogramDataset


class AbstractDataModule(pl.LightningDataModule):
    """
    Basic DataModule to load data during experiments, designed for convolutional networks.
    """
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
                 source_name: str = 'record',
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
                 train_dataset: type or partial = None,
                 window_length: int = 1
                 ):
        """
        Args:
            df_path: Path to stored dataframe
            width: Width of sample from dataset
            height: Height of sample from dataset
            channels: Channels of sample from dataset
            num_classes: Number of classes that sample can be classified as such.
            train_transforms: Sequence of transforms for training dataset.
            val_transforms: Sequence of transforms for validation dataset.
            test_transforms: Sequence of transforms for test dataset.
            train_vs_rest_size: Part of the whole data that should be used for training. Not important for crossvalidation.
            val_vs_test_size: Part of the data unused for training that should be used for validation. Not important for crossvalidation.
            source_name: Name of column in dataframe with data
            target_name: Name of column in dataframe with labels
            series_name: Name of column in dataframe with series number
            subject_name: Name of column in dataframe with subject number
            batch_size: Size of single batch of data.
            num_workers: Number of workers used for loading data
            shuffle_train: Should training data be shuffled for every epoch
            seed: Seed value for all random generators taking part in process of loading data
            k_folds: Number of folds to which data should be split. If less than 3 then data is split once based on train_vs_rest_size and val_vs_test_size
            dataset: Type of dataset that will be used for validation and testing. If train_dataset isn't defined, it will also be used for training.
            split_method: Method for splitting data. Can be set as 'equal', 'trials', 'subjects' or None.
                          If None, then data series are split randomly.
                          If 'equal', then each split will have the same subjects and gestures, in equal amount, but different trialls in them.
                          If 'trials', then each split will have the same subjects but different trialls in them.
                          If 'subjects', then each split will have different subjects in it.
            train_dataset: Type of dataset that will be used for training. If isn't defined, 'dataset' will also be used for training.
            window_length: Amount of samples from original dataset to compress into one.
        """
        super(AbstractDataModule, self).__init__()
        self.window_length = window_length
        self.mean = None
        self.std = None
        self.df_path: os.PathLike = df_path
        self.data: pd.DataFrame = pd.DataFrame()
        self.subjects = []
        self.series = []
        self.targets = []
        self.width = width
        self.height = height
        self.channels = channels
        self.num_classes = num_classes
        self.k_folds = k_folds
        if self.k_folds < 3:
            self.train_vs_rest_size = train_vs_rest_size
            self.val_vs_test_size = val_vs_test_size
        else:
            self.k_folds = k_folds
            self.fold = 0
            self.folds = []
        self.train_transforms: Compose = train_transforms
        self.val_transforms: Compose = val_transforms
        self.test_transforms: Compose = test_transforms
        self.source_name: str = source_name
        self.target_name: str = target_name
        self.series_name: str = series_name
        self.subject_name: str = subject_name
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.shuffle_train: bool = shuffle_train
        self.splits: Dict[str, Index] = {}
        self.seed: int = randint(0, 2**32 - 1) if seed is None else seed
        self.dataset: type = dataset
        self.train_dataset: type or partial = dataset if train_dataset is None else train_dataset
        random.seed(self.seed)

        self.split_method = split_method

    def prepare_data(self) -> None:
        self.data = pd.read_pickle(self.df_path)
        self.subjects = list(self.data[self.subject_name].unique())
        self.series = list(self.data[self.series_name].unique())
        self.targets = list(self.data[self.target_name].unique())

    def setup(self, stage: Optional[str] = None) -> None:
        self.split_data()
        self.calculate_mean_std()

    def split_data(self) -> None:
        """
        Split data into proper train, validate and test splits. If number of folds is set to more than 2 it will prepare
        all folds at the first time when this method will be called.
        Returns:

        """
        if self.k_folds < 2:
            if not self.splits:
                train_series = self.get_random_series(self.data, proportion=self.train_vs_rest_size)
                non_train_data = self.data.loc[~self.data[self.series_name].isin(train_series)]
                val_series = self.get_random_series(non_train_data, proportion=self.val_vs_test_size)
                test_series = list(set(self.series) - set(train_series) - set(val_series))
                self.splits['train'] = self.data.index[self.data[self.series_name].isin(train_series)]
                self.splits['val'] = self.data.index[self.data[self.series_name].isin(val_series)]
                self.splits['test'] = self.data.index[self.data[self.series_name].isin(test_series)]
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

    def calculate_mean_std(self) -> None:
        """Calculate mean value and standard deviation from training dataset.
        Then set all Normalization transforms mean and std to the calculated ones"""
        train_values = np.stack([item for _, item in self.data.iloc[self.splits['train']][self.source_name].iteritems()])
        self.mean = np.mean(train_values)
        self.std = np.std(train_values)
        norm_idx = None
        for idx, tr in enumerate(self.train_transforms.transforms):
            if type(tr) is Normalize:
                norm_idx = idx
        if norm_idx is not None:
            self.train_transforms.transforms[norm_idx].mean = self.mean
            self.train_transforms.transforms[norm_idx].std = self.std

        norm_idx = None
        for idx, tr in enumerate(self.val_transforms.transforms):
            if type(tr) is Normalize:
                norm_idx = idx
        if norm_idx is not None:
            self.val_transforms.transforms[norm_idx].mean = self.mean
            self.val_transforms.transforms[norm_idx].std = self.std

        norm_idx = None
        for idx, tr in enumerate(self.test_transforms.transforms):
            if type(tr) is Normalize:
                norm_idx = idx
        if norm_idx is not None:
            self.test_transforms.transforms[norm_idx].mean = self.mean
            self.test_transforms.transforms[norm_idx].std = self.std

    def get_random_series(self, data: pd.DataFrame, proportion: float) -> list or None:
        """
        Get subset of data which size is equal to given dataset size multiply by proportion parm
        Subset creation method will depend on split_method parameter value:
        - If None, then data series are split randomly.
        - If 'equal', then each split will have the same subjects and gestures, in equal amount, but different trialls in them.
        - If 'trials', then each split will have the same subjects but different trialls in them.
        - If 'subjects', then each split will have different subjects in it.
        Args:
            data: Dataset from which sub-dataset will be created
            proportion: Value in range (0. , 1.). Dictates how big part of dataset will be chosen for sub-dataset.
        """
        if self.split_method == 'equal':
            full_series = []
            for subject in self.subjects:
                for target in self.targets:
                    series = list(data.loc[data[self.subject_name] == subject].loc[data[self.target_name] == target]
                                  [self.series_name].unique())
                    full_series += random.sample(series, math.floor(len(series) * proportion))
            return full_series
        elif self.split_method == 'trials':
            full_series = []
            for subject in self.subjects:
                series = list(data.loc[data[self.subject_name] == subject][self.series_name].unique())
                full_series += random.sample(series, math.floor(len(series) * proportion))
            return full_series
        elif self.split_method == 'subject':
            subject_ids = list(data[self.subject_name].unique())
            subjects = random.sample(subject_ids, math.floor(len(subject_ids) * proportion))
            series = data.loc[data[self.subject_name].isin(subjects)][self.series_name].unique()
            return series
        series = list(data[self.series_name].unique())
        return random.sample(series, math.floor(len(series) * proportion))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(data_frame=self.data.iloc[self.splits['train']].reset_index(),
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
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['val']].reset_index(),
                         transform=self.val_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['test']].reset_index(),
                         transform=self.test_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name,
                         window_length=self.window_length),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def get_data_parameters(self) -> Dict:
        """
        Get parameters describing dataset stored by datamodule.
        Returns: Dictionary containing:
        - 'num_classes',
        - 'input_width',
        - 'input_height',
        - 'channels'
        """
        return {'num_classes': self.num_classes, 'input_width': self.width, 'input_height': self.height,
                'channels': self.channels}

    def set_fold(self, fold: int) -> None:
        """
        Set number of current fold. Makes it always less than k_fold parameter.
        Args:
            fold: Proposed number of fold
        """
        if self.k_folds > 2:
            self.fold = fold % self.k_folds

    def next_fold(self) -> None:
        """
        Set current fold to the next one in order.
        """
        if self.k_folds > 2:
            self.set_fold(self.fold + 1)

    def get_splits_series(self) -> Dict:
        """
        Get train, validation and test splits.
        Returns: dictionary with keys:
        - train
        - val
        - test
        """
        return {'train': self.data.iloc[self.splits['train']][self.series_name].unique(),
                'val': self.data.iloc[self.splits['val']][self.series_name].unique(),
                'test': self.data.iloc[self.splits['test']][self.series_name].unique()}

    def set_window_length(self, new_length: int) -> None:
        """
        Change length of the window
        Args:
            new_length: new length of the window
        """
        self.window_length = new_length
