import os
from typing import Dict

from torch import nn
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_myoarmband, prepare_dataframe_dataset, SequenceDataModule, SequenceDataset, AbstractDataset
from definitions import PKL_FOLDER


class MyoArmbandDataModule(SequenceDataModule):
    def __init__(self,
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
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None,
                 dataset: type = SequenceDataset,
                 feature_extraction_dataset: type = AbstractDataset,
                 window_length: int = 1,
                 window_step: int = 1):
        df_path = os.path.join(PKL_FOLDER, 'MyoArmband', 'MyoArmband.pkl')
        if not os.path.isfile(df_path):
            prepare_myoarmband(prepare_dataframe_dataset, PKL_FOLDER)
        super(MyoArmbandDataModule, self).__init__(
            df_path,
            1,
            8,
            1,
            7,
            feature_extraction,
            splits_series,
            feature_extraction_transforms,
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
            dataset,
            feature_extraction_dataset,
            window_length,
            window_step)
