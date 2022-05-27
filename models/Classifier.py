from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch import Tensor
import torch.nn as nn


class Classifier(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float = 1e-3,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.Conv2d(64, 64, (1, 1), (1, 1)),
            nn.Conv2d(64, 64, (1, 1), (1, 1)),
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 128),
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )
        self.lr = lr
        self.optimizer = optimizer
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), lr=self.lr)

    def training_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss = self.__step_basics(train_batch)
        logs = {'train_loss': loss}
        return {"loss": loss, 'log': logs}

    def validation_step(self, val_batch: tuple[Tensor, Tensor], batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss = self.__step_basics(val_batch)
        return {"val_loss": loss}

    def __step_basics(self, batch: tuple[Tensor, Tensor]):
        x, y = batch
        if torch.cuda.is_available():
            x.cuda()
            y.cuda()
        outputs = self.forward(x)
        return self.criterion(outputs, y)
