import os
from torchvision.transforms import Compose, ToTensor

from datasets.data_import import prepare_ninapro
from datasets.abstract_data_module import AbstractDataModule


class NinaProDataModule(AbstractDataModule):
    def __init__(self,
                 train_transforms: Compose = ToTensor(),
                 val_transforms: Compose = ToTensor(),
                 test_transforms: Compose = ToTensor(),
                 train_vs_rest_size: float = 0.8,
                 val_vs_test_size: float = 0.5,
                 source_path_name: str = 'path',
                 target_name: str = 'label',
                 batch_size: int = 12,
                 num_workers: int = 8,
                 shuffle_train: bool = True,
                 seed: int = None):
        csv_path = os.path.join('..', '..', 'Data', 'NinaPro', 'NinaPro.csv')
        if not os.path.isfile(csv_path):
            prepare_ninapro()
        super(NinaProDataModule, self).__init__(
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
