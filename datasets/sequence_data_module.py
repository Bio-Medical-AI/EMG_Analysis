import os
from functools import partial
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torch.nn.functional import pad
from datasets import AbstractDataModule, AbstractDataset
from datasets.sequence_dataset import SequenceDataset


class SequenceDataModule(AbstractDataModule):
    """
    DataModule to load data during experiments, designed for Recurrent Networks (RNN, CRNN)
    """
    def __init__(self,
                 df_path: os.PathLike or str,
                 width: int,
                 height: int,
                 channels: int,
                 num_classes: int,
                 feature_extraction: nn.Module,
                 splits_series: Dict,
                 feature_extraction_transforms: Compose = ToTensor(),
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
                 num_workers: int = 4,
                 shuffle_train: bool = True,
                 seed: int = None,
                 dataset: type or partial = SequenceDataset,
                 feature_extraction_dataset: type or partial = AbstractDataset,
                 train_dataset: type or partial = None,
                 window_length: int = 1,
                 window_step: int = None
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
            dataset: Type of dataset that will be used for validation and testing. If train_dataset isn't defined, it will also be used for training.
            train_dataset: Type of dataset that will be used for training. If isn't defined, 'dataset' will also be used for training.
            window_length: Amount of samples from original dataset to compress into one.
        """
        self.feature_extraction_dataset = feature_extraction_dataset
        self.feature_extraction: nn.Module = feature_extraction
        self.window_step: int = window_step
        self.splits_series = splits_series
        self.feature_extraction_transforms = feature_extraction_transforms
        self.length_name = 'length'
        super(SequenceDataModule, self).__init__(
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
            0,
            dataset,
            'default',
            train_dataset,
            window_length
        )

    def prepare_data(self) -> None:
        super(SequenceDataModule, self).prepare_data()
        self.window_length = min([self.data[self.series_name].value_counts().max(), self.window_length])
        self.window_step = self.window_step if self.window_step is not None else self.window_length
        loader = DataLoader(
            self.feature_extraction_dataset(data_frame=self.data,
                                            transform=self.feature_extraction_transforms,
                                            source_name=self.source_name,
                                            target_name=self.target_name,
                                            series_name=self.series_name),
            batch_size=int(self.window_length),
            num_workers=self.num_workers
        )
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        self.feature_extraction.to(device)
        self.data[self.source_name] = [i for i in torch.cat(
            [self.feature_extraction(batch['data'].to(device)).detach().cpu()
             for batch in tqdm(loader, 'feature extraction')])]
        all_sources = []
        all_targets = []
        all_series = []
        all_subjects = []
        all_lengths = []
        for s in tqdm(self.series, "padding series"):
            records = self.data[self.data[self.series_name] == s].reset_index()
            sources = records[self.source_name]
            targets = records[self.target_name]
            series = records.at[0, self.series_name]
            subjects = records.at[0, self.subject_name]
            iters = (len(records) - self.window_length) // self.window_step + 1

            all_sources += [
                torch.stack(sources[i*self.window_step:i*self.window_step+self.window_length].tolist()).numpy() for i in
                range(iters)]
            all_targets += [np.array(targets[i*self.window_step:i*self.window_step+self.window_length].tolist())
                            for i in range(iters)]
            all_series += [series for _ in range(iters)]
            all_subjects += [subjects for _ in range(iters)]
            all_lengths += [self.window_length for _ in range(iters)]
            if (len(records) - self.window_length) % self.window_step != 0:
                length = len(records) - iters * self.window_step
                all_sources.append(
                    pad(
                        torch.stack(
                            sources[iters * self.window_step:len(records)].tolist()
                        ),
                        (0, 0, 0, self.window_length - length), "constant", 0.).numpy())
                all_targets.append(
                    pad(
                        torch.Tensor(
                            targets[iters * self.window_step:len(records)].tolist()
                        ),
                        (0, self.window_length - length), "constant", 0.).numpy())
                all_series.append(series)
                all_subjects.append(subjects)
                all_lengths.append(length)

        self.data = pd.DataFrame({self.source_name: all_sources,
                                  self.target_name: all_targets,
                                  self.series_name: all_series,
                                  self.subject_name: all_subjects,
                                  self.length_name: all_lengths})

    def setup(self, stage: Optional[str] = None) -> None:
        self.split_data()

    def split_data(self) -> None:
        """
        Prepares training, validation and test splits.
        """
        self.splits['train'] = self.data.index[self.data[self.series_name].isin(self.splits_series['train'])]
        self.splits['val'] = self.data.index[self.data[self.series_name].isin(self.splits_series['val'])]
        self.splits['test'] = self.data.index[self.data[self.series_name].isin(self.splits_series['test'])]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset(data_frame=self.data.iloc[self.splits['train']].reset_index(),
                               transform=self.train_transforms,
                               source_name=self.source_name,
                               target_name=self.target_name,
                               series_name=self.series_name),
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
                         series_name=self.series_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset(data_frame=self.data.iloc[self.splits['test']].reset_index(),
                         transform=self.test_transforms,
                         source_name=self.source_name,
                         target_name=self.target_name,
                         series_name=self.series_name),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )