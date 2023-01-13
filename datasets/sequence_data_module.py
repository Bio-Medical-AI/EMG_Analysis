import os
from functools import partial

import numpy as np
import pandas as pd
from torchvision.transforms import Compose, ToTensor

from datasets import AbstractDataModule, AbstractDataset


class SequenceDataModule(AbstractDataModule):
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
                 dataset: type or partial = AbstractDataset,
                 split_method: str = 'default',
                 train_dataset: type or partial = None
                 ):

        self.window_length: int = batch_size
        self.window_step: int = batch_size

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
            1,
            num_workers,
            shuffle_train,
            seed,
            k_folds,
            dataset,
            split_method,
            train_dataset
        )

    def prepare_data(self) -> None:
        super(SequenceDataModule, self).prepare_data()
        self.window_length = min([self.data[self.series_name].value_counts().max(), self.window_length])
        self.window_step = self.window_length
        all_sources = []
        all_targets = []
        all_series = []
        all_subjects = []
        for s in self.series:
            records = self.data[self.data[self.series_name] == s].reset_index()
            sources = records[self.source_name]
            targets = records.at[0, self.target_name]
            series = records.at[0, self.series_name]
            subjects = records.at[0, self.subject_name]
            iters = (len(records) - self.window_length) // self.window_step + 1

            all_sources += [
                np.stack(sources[i*self.window_step:i*self.window_step+self.window_length]) for i in range(iters)]
            all_targets += [targets for _ in range(iters)]
            all_series += [series for _ in range(iters)]
            all_subjects += [subjects for _ in range(iters)]
            if (len(records) - self.window_length) % self.window_step != 0:
                all_sources.append(
                    np.stack(sources[iters * self.window_step:len(records)]))
                all_targets.append(targets)
                all_series.append(series)
                all_subjects.append(subjects)

        self.data = pd.DataFrame({self.source_name: all_sources,
                                  self.target_name: all_targets,
                                  self.series_name: all_series,
                                  self.subject_name: all_subjects})
