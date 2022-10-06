import os

import torchvision
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_capgmyo, prepare_frame_dataset, AbstractDataModule
from utils.transforms import LOAD_NDARRAY
from definitions import FRAMES_FOLDER


class CapgMyoDataModule(AbstractDataModule):
    def __init__(self,
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
                 seed: int = None,
                 k_folds: int = 0):
        df_path = os.path.join(FRAMES_FOLDER, 'CapgMyo', 'CapgMyo.csv')
        if not os.path.isfile(df_path):
            prepare_capgmyo(prepare_frame_dataset, FRAMES_FOLDER)
        super(CapgMyoDataModule, self).__init__(
            df_path,
            16,
            8,
            8,
            torchvision.transforms.Compose([LOAD_NDARRAY, train_transforms]),
            torchvision.transforms.Compose([LOAD_NDARRAY, val_transforms]),
            torchvision.transforms.Compose([LOAD_NDARRAY, test_transforms]),
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
