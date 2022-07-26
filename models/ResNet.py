import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torchmetrics.functional import accuracy
import torchvision.models as models


class ResNet(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float = 1e-3,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.Adam):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.lr = lr
        self.optimizer = optimizer
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), lr=self.lr)

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


class ResNetTest(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 lr: float = 1e-3,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.Adam):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.lr = lr
        self.optimizer = optimizer
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer(self.parameters(), lr=self.lr)

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
