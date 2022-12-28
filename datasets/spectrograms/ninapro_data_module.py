import os

from torchvision.transforms import Compose, ToTensor
from datasets import prepare_ninapro, prepare_dataframe_dataset, SpectrogramDataModule, SpectrogramDataset
from definitions import SERIES_FOLDER


class NinaProDataModule(SpectrogramDataModule):
    def __init__(self,
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
                 dataset: type = SpectrogramDataset,
                 split_method: str = 'default',
                 series_length: int = 10,
                 window_step: int = 1):
        df_path = os.path.join(SERIES_FOLDER, 'NinaPro', 'NinaPro.pkl')
        if not os.path.isfile(df_path):
            prepare_ninapro(prepare_dataframe_dataset, SERIES_FOLDER)
        super(NinaProDataModule, self).__init__(
            df_path,
            series_length,
            10,
            1,
            52,
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
            k_folds,
            dataset,
            split_method,
            series_length,
            window_step)
