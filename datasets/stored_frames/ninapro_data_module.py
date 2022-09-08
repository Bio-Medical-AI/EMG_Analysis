import os

import torchvision
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_ninapro, prepare_frame_dataset, AbstractDataModule
from utils.transforms import LOAD_NDARRAY


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
            prepare_ninapro(prepare_frame_dataset,
                            os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data'))
        super(NinaProDataModule, self).__init__(
            csv_path,
            torchvision.transforms.Compose([LOAD_NDARRAY, train_transforms]),
            torchvision.transforms.Compose([LOAD_NDARRAY, val_transforms]),
            torchvision.transforms.Compose([LOAD_NDARRAY, test_transforms]),
            train_vs_rest_size,
            val_vs_test_size,
            source_path_name,
            target_name,
            batch_size,
            num_workers,
            shuffle_train,
            seed)
