import torch
import numpy as np
import pandas as pd
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class AbstractDataset(Dataset):
    def __init__(self,
                 data_frame: pd.DataFrame,
                 transform: Compose = ToTensor(),
                 source_path_name: str = 'path',
                 target_name: str = 'label'):
        self.records: pd.Series = data_frame[source_path_name]
        self.labels: pd.Series = data_frame[target_name]
        self.transform = transform
        self.samples_amount = data_frame.shape[0]

    def __getitem__(self, index: int):
        if self.transform is None:
            return np.load(self.records.iloc[index]), self.labels[index]
        # try:
        #     data = self.transform(np.load(self.records.iloc[index])).float()
        # except:
        #     raise Exception(self.records.iloc[index])
        data = self.transform(np.load(self.records.iloc[index])).float()
        label = torch.tensor(self.labels.iloc[index]).long()
        return data, label

    def __len__(self) -> int:
        return self.samples_amount
