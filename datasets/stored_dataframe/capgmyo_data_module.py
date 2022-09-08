import os

import pandas as pd
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_capgmyo, prepare_dataframe_dataset, AbstractDataModule


class CapgMyoDataModule(AbstractDataModule):
    def __init__(self,
                 train_transforms: Compose = ToTensor(),
                 val_transforms: Compose = ToTensor(),
                 test_transforms: Compose = ToTensor(),
                 train_vs_rest_size: float = 0.8,
                 val_vs_test_size: float = 0.5,
                 source_path_name: str = 'record',
                 target_name: str = 'label',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None):
        csv_path = os.path.join('..', '..', 'Data_pkl', 'CapgMyo', 'CapgMyo.pkl')
        if not os.path.isfile(csv_path):
            prepare_capgmyo(prepare_dataframe_dataset,
                            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data_pkl'))
        super(CapgMyoDataModule, self).__init__(
            csv_path,
            train_transforms,
            val_transforms,
            test_transforms,
            train_vs_rest_size,
            val_vs_test_size,
            source_path_name,
            target_name,
            batch_size,
            num_workers,
            shuffle_train,
            seed)

    def prepare_data(self) -> None:
        self.data: pd.DataFrame = pd.read_pickle(self.csv_path)
