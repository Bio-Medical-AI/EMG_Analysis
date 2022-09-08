from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy, f1_score, precision, specificity


class Classifier(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.AdamW,
                 lr_scheduler: type(torch.optim.lr_scheduler) = torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model',
                                          'criterion',
                                          'lr_lambda',
                                          'lr_scheduler',
                                          'optimizer'])
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.lr_lambda = kwargs.get('lr_lambda', None)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> dict[str, dict[str, LambdaLR | None] | Any]:
        optim_kwargs = self.hparams.optim_kwargs if 'optim_kwargs' in self.hparams else {}
        optimizer = self.optimizer(self.model.parameters(), **optim_kwargs)

        sched_kwargs = self.hparams.sched_kwargs if 'sched_kwargs' in self.hparams else {}
        if self.lr_lambda is not None:
            sched_kwargs['lr_lambda'] = self.lr_lambda
        scheduler = self.lr_scheduler(optimizer, **sched_kwargs)

        monitor = self.hparams.monitor if 'monitor' in self.hparams else None

        lr_scheduler_config = {
            "scheduler": scheduler,
            "monitor": monitor}
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config}

    def training_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, f1, prec, spec = self._step(train_batch)

        logs = {'loss': loss}
        self.log_dict(logs)

        logs_2 = {'train_accuracy': acc,
                  'train_f1': f1,
                  'train_precision': prec,
                  'train_specificity': spec
                  }
        self.log_dict(logs_2, on_step=False, on_epoch=True)
        logs_2['loss'] = loss
        return logs_2

    def validation_step(self, val_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, f1, prec, spec = self._step(val_batch)

        logs = {'loss': loss,
                'val_accuracy': acc,
                'val_f1': f1,
                'val_precision': prec,
                'val_specificity': spec
                }
        self.log_dict(logs)

        return logs

    def test_step(self, test_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss, acc, f1, prec, spec = self._step(test_batch)
        logs = {'loss': loss,
                'test_accuracy': acc,
                'test_f1': f1,
                'test_precision': prec,
                'test_specificity': spec
                }
        self.log_dict(logs)
        return logs

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)

    def _step(self, batch: tuple[Tensor, Tensor]) -> tuple[float, Tensor, Tensor, Tensor, Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, average='micro')
        f1 = f1_score(preds, y, average='micro')
        prec = precision(preds, y, average='micro')
        spec = specificity(preds, y, average='micro')
        return loss, acc, f1, prec, spec
