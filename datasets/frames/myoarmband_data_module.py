import os
from torchvision.transforms import Compose, ToTensor

from datasets import prepare_myoarmband, prepare_dataframe_dataset, AbstractDataModule, SequenceDataset
from definitions import PKL_FOLDER


class MyoArmbandDataModule(AbstractDataModule):
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
                 split_method: str = 'subject',
                 train_dataset: type = None,
                 window_length: int = 1):
        df_path = os.path.join(PKL_FOLDER, 'MyoArmband', 'MyoArmband.pkl')
        if not os.path.isfile(df_path):
            prepare_myoarmband(prepare_dataframe_dataset, PKL_FOLDER)
        super(MyoArmbandDataModule, self).__init__(
            df_path,
            window_length,
            8,
            1,
            7,
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
