import os

import pandas as pd
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_csl, prepare_dataframe_dataset, AbstractDataModule
from definitions import SERIES_FOLDER


class CslDataModule(AbstractDataModule):
    def __init__(self,
                 train_transforms: Compose = ToTensor(),
                 val_transforms: Compose = ToTensor(),
                 test_transforms: Compose = ToTensor(),
                 train_vs_rest_size: float = 0.8,
                 val_vs_test_size: float = 0.5,
                 source_path_name: str = 'series',
                 target_name: str = 'label',
                 group_name: str = 'series',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None,
                 k_folds: int = 0):
        df_path = os.path.join(SERIES_FOLDER, 'csl-hdemg', 'csl-hdemg.pkl')
        if not os.path.isfile(df_path):
            prepare_csl(prepare_dataframe_dataset, SERIES_FOLDER)
        super(CslDataModule, self).__init__(
            df_path,
            7,
            24,
            27,
            train_transforms,
            val_transforms,
            test_transforms,
            train_vs_rest_size,
            val_vs_test_size,
            source_path_name,
            target_name,
            group_name,
            batch_size,
            num_workers,
            shuffle_train,
            seed,
            k_folds)

    def prepare_data(self) -> None:
        self.data: pd.DataFrame = pd.read_pickle(self.df_path)
