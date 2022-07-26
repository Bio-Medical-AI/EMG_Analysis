from typing import Any, Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy


class Classifier(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.Adam,
                 lr_scheduler_lambda: Callable[[int], float] =
                    lambda epoch: 1 if epoch < 17 else (0.1 if epoch < 25 else 0.01)):
        super().__init__()
        if lr_scheduler_lambda is None:
            lr_scheduler_lambda = lambda epoch: 1
        self.model = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.weight_initialization()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler_lambda = lr_scheduler_lambda
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.criterion = nn.CrossEntropyLoss()

    def weight_initialization(self):
        for layer in self.model:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, dict[str, LambdaLR | None] | Any]:
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scheduler_lambda),
            "monitor": None}
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        outputs, y = self.__step_basics(train_batch)
        loss = self.criterion(outputs, y)
        logs = {'loss': loss}
        self.log_dict(logs)
        return logs

    def validation_step(self, val_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        outputs, y = self.__step_basics(val_batch)
        loss = self.criterion(outputs, y)
        acc = accuracy(outputs, y)
        logs = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(logs)
        return logs

    def test_step(self, test_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        outputs, y = self.__step_basics(test_batch)
        loss = self.criterion(outputs, y)
        acc = accuracy(outputs, y)
        logs = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(logs)
        return logs

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)

    def __step_basics(self, batch: tuple[Tensor, Tensor]):
        x, y = batch
        if torch.cuda.is_available():
            x.cuda()
            y.cuda()
        outputs = self.forward(x)
        return outputs, y
