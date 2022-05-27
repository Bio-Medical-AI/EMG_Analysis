import os

import pandas as pd
from torchvision.transforms import Compose

from datasets.abstract_dataset import AbstractDataset


class NinaPro(AbstractDataset):
    def __init__(self, transform: Compose = None):
        super().__init__(pd.read_csv(os.path.join('..', '..', 'Data', 'NinaPro', 'NinaPro.csv')), transform)
