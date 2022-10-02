from typing import Any, Dict

import pandas as pd
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
                 time_window: int = 150,
                 time_step: int = 10,
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
        self.time_window = time_window
        self.time_step = time_step
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

    def training_step(self, train_batch: dict[str, Tensor | Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(train_batch)

        logs = {'loss': results['loss']}
        self.log_dict(logs)

        logs_2 = {'train_accuracy': results['acc'],
                  'train_f1': results['f1'],
                  'train_precision': results['prec'],
                  'train_specificity': results['spec']
                  }
        self.log_dict(logs_2, on_step=False, on_epoch=True)
        logs_2['loss'] = results['loss']
        return logs_2

    def validation_step(self, val_batch: dict[str, Tensor | Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(val_batch)

        logs = {'loss': results['loss'],
                'val_accuracy': results['acc'],
                'val_f1': results['f1'],
                'val_precision': results['prec'],
                'val_specificity': results['spec']
                }
        self.log_dict(logs)
        logs['preds'] = results['preds']
        logs['labels'] = results['labels']
        logs['group'] = results['group']
        logs['index'] = results['index']
        return logs

    def validation_epoch_end(self, validation_step_outputs: list[dict[str, Tensor | Any]]):
        results = self._epoch_end(validation_step_outputs)

        logs = {'val_majority_voting_accuracy': results['acc'],
                'val_majority_voting_f1': results['f1'],
                'val_majority_voting_precision': results['prec'],
                'val_majority_voting_specificity': results['spec']
                }
        self.log_dict(logs)

    def test_step(self, test_batch: dict[str, Tensor | Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(test_batch)
        logs = {'loss': results['loss'],
                'test_accuracy': results['acc'],
                'test_f1': results['f1'],
                'test_precision': results['prec'],
                'test_specificity': results['spec']
                }
        self.log_dict(logs)
        logs['preds'] = results['preds']
        logs['labels'] = results['labels']
        logs['group'] = results['group']
        logs['index'] = results['index']
        return logs

    def test_epoch_end(self, test_step_outputs: list[dict[str, Tensor | Any]]):
        results = self._epoch_end(test_step_outputs)

        logs = {'test_majority_voting_accuracy': results['acc'],
                'test_majority_voting_f1': results['f1'],
                'test_majority_voting_precision': results['prec'],
                'test_majority_voting_specificity': results['spec']
                }
        self.log_dict(logs)

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)

    def _step(self, batch: dict[str, Tensor | Any]) -> dict[str, Tensor | Any]:
        x = batch['data']
        y = batch['label']
        group = batch['group']
        index = batch['index']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, average='micro')
        f1 = f1_score(preds, y, average='micro')
        prec = precision(preds, y, average='micro')
        spec = specificity(preds, y, average='micro')
        return {'loss': loss,
                'acc': acc,
                'f1': f1,
                'prec': prec,
                'spec': spec,
                'preds': preds,
                'labels': y,
                'group': group,
                'index': index}

    def _moving_average(self, df: pd.DataFrame):
        if df.shape[0] >= self.time_window:
            preds = []
            labels = []
            for i in range((df.shape[0] - self.time_window) // self.time_step):
                tmp = df.iloc[(i * self.time_step):(i * self.time_step + self.time_window)]
                preds.append(tmp['preds'].mode()[0].item())
                labels.append(tmp['labels'].mode()[0].item())
            return pd.DataFrame({'preds': preds, 'labels': labels})
        else:
            return pd.DataFrame({'preds': df['preds'].mode()[0].item(), 'labels': df['labels'].mode()[0].item()})

    def _epoch_end(self, step_outputs: list[dict[str, Tensor | Any]]) -> dict[str, Tensor | Any]:
        full_preds = self._connect_epoch_results(step_outputs, 'preds')
        full_labels = self._connect_epoch_results(step_outputs, 'labels')
        group = self._connect_epoch_results(step_outputs, 'group')
        index = self._connect_epoch_results(step_outputs, 'index')
        df = pd.DataFrame({'preds': full_preds,
                           'labels': full_labels,
                           'group': group,
                           'index': index}).sort_values(by=['index'])

        tmp_df = pd.DataFrame(columns=['preds', 'labels'])
        for series in df['group'].unique().tolist():
            tmp_df = tmp_df.append(self._moving_average(df.loc[df['group'] == series]), ignore_index=True)

        preds = torch.tensor(tmp_df['preds'].values.tolist())
        labels = torch.tensor(tmp_df['labels'].values.tolist())
        acc = accuracy(preds, labels, average='micro')
        f1 = f1_score(preds, labels, average='micro')
        prec = precision(preds, labels, average='micro')
        spec = specificity(preds, labels, average='micro')
        return {'acc': acc, 'f1': f1, 'prec': prec, 'spec': spec}

    def _connect_epoch_results(self, step_outputs: list[dict[str, Tensor | Any]], key: str):
        to_concat = []
        for output in step_outputs:
            to_concat.append(output[key].detach().cpu())
        return torch.cat(to_concat)
