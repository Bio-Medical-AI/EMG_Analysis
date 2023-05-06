import os
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_ninapro, prepare_dataframe_dataset, AbstractDataModule, SequenceDataset
from definitions import PKL_FOLDER


class NinaProDataModule(AbstractDataModule):
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
                 dataset: type = SequenceDataset,
                 split_method: str = 'equal',
                 train_dataset: type = None,
                 window_length: int = 1):
        """
        Args:
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
        df_path = os.path.join(PKL_FOLDER, 'NinaPro', 'NinaPro.pkl')
        if not os.path.isfile(df_path):
            prepare_ninapro(prepare_dataframe_dataset, PKL_FOLDER)
        super(NinaProDataModule, self).__init__(
            df_path,
            window_length,
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
            train_dataset,
            window_length)

    def set_window_length(self, new_length: int) -> None:
        super().set_window_length(new_length)
        self.width = new_length
