from typing import Any, Dict, List

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
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

    def configure_optimizers(self) -> Dict[str, Dict[str, LambdaLR or None] or Any]:
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

    def training_step(self, train_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(train_batch)
        logs = {'loss': results['loss']}
        self.log_dict(logs)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results = self._epoch_end(outputs)
        self.log_dict({
            'train_accuracy': results['acc'],
            'train_f1': results['f1'],
            'train_speccificity': results['spec'],
            'train_precision': results['prec']})

    def validation_step(self, val_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(val_batch)
        return results

    def validation_epoch_end(self, validation_step_outputs: EPOCH_OUTPUT):
        results = self._epoch_end(validation_step_outputs)
        results = self._eval_epoch_end(results)

        logs = {'val_majority_voting_accuracy': results['majority_acc'],
                'val_majority_voting_f1': results['majority_f1'],
                'val_majority_voting_precision': results['majority_prec'],
                'val_majority_voting_specificity': results['majority_spec'],
                'val_precision': results['prec'],
                'val_specificity': results['spec'],
                'val_accuracy': results['acc'],
                'val_f1': results['f1']
                }
        self.log_dict(logs)

    def test_step(self, test_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(test_batch)
        return results

    def test_epoch_end(self, test_step_outputs: EPOCH_OUTPUT):
        results = self._epoch_end(test_step_outputs)
        results = self._eval_epoch_end(results)

        logs = {'test_majority_voting_accuracy': results['majority_acc'],
                'test_majority_voting_f1': results['majority_f1'],
                'test_majority_voting_precision': results['majority_prec'],
                'test_majority_voting_specificity': results['majority_spec'],
                'test_precision': results['prec'],
                'test_specificity': results['spec'],
                'test_accuracy': results['acc'],
                'test_f1': results['f1']
                }
        self.log_dict(logs)
        self.trainer.logger.finalize('success')

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)

    def _step(self, batch: Dict[str, Tensor or Any]) -> Dict[str, Tensor or Any]:
        x = batch['data']
        y = batch['label']
        series = batch['series']
        index = batch['index']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss,
                'preds': preds,
                'labels': y,
                'series': series,
                'index': index}

    def _moving_average(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[0] >= self.time_window:
            preds = []
            labels = []
            for i in range((df.shape[0] - self.time_window) // self.time_step + 1):
                tmp = df.iloc[(i * self.time_step):(i * self.time_step + self.time_window)]
                preds.append(tmp['preds'].values.tolist())
                labels.append(tmp['labels'].head(1).item())
            return pd.DataFrame({'preds': torch.mode(torch.Tensor(preds))[0].tolist(), 'labels': labels})
        else:
            return pd.DataFrame({'preds': [df['preds'].mode()[0].item()], 'labels': [df['labels'].mode()[0].item()]})

    def _epoch_end(self, step_outputs: EPOCH_OUTPUT) -> STEP_OUTPUT:
        preds = self._connect_epoch_results(step_outputs, 'preds')
        labels = self._connect_epoch_results(step_outputs, 'labels')
        series = self._connect_epoch_results(step_outputs, 'series')
        index = self._connect_epoch_results(step_outputs, 'index')
        loss = [step['loss'] for step in step_outputs]
        acc = accuracy(preds, labels, average='micro')
        f1 = f1_score(preds, labels, average='macro', num_classes=self.model.num_classes)
        prec = precision(preds, labels, average='macro', num_classes=self.model.num_classes)
        spec = specificity(preds, labels, average='macro', num_classes=self.model.num_classes)
        return {'loss': loss,
                'preds': preds,
                'labels': labels,
                'series': series,
                'index': index,
                'acc': acc,
                'f1': f1,
                'prec': prec,
                'spec': spec}

    def _eval_epoch_end(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        tmp_dict = {
            'preds': step_outputs['preds'],
            'labels': step_outputs['labels'],
            'series': step_outputs['series'],
            'index': step_outputs['index'],
        }
        print('sort -------------------------------------------')
        print('sort -------------------------------------------')
        df = pd.DataFrame(tmp_dict).sort_values(by=['index'])
        print('end sort -------------------------------------------')
        print('end sort -------------------------------------------')
        tmp_df = pd.DataFrame(columns=['preds', 'labels'])
        print('moving avg -------------------------------------------')
        print('moving avg -------------------------------------------')
        for series in df['series'].unique().tolist():
            tmp_df = tmp_df.append(self._moving_average(df.loc[df['series'] == series]), ignore_index=True)
        print('end moving avg -------------------------------------------')
        print('end moving avg -------------------------------------------')
        print(tmp_df)

        majority_preds = torch.tensor(tmp_df['preds'].values.tolist())
        majority_labels = torch.tensor(tmp_df['labels'].values.tolist())
        majority_acc = accuracy(majority_preds, majority_labels, average='micro')
        majority_f1 = f1_score(majority_preds, majority_labels, average='macro', num_classes=self.model.num_classes)
        majority_prec = precision(majority_preds, majority_labels, average='macro', num_classes=self.model.num_classes)
        majority_spec = specificity(majority_preds, majority_labels, average='macro',
                                    num_classes=self.model.num_classes)
        return {'loss': step_outputs['loss'],
                'acc': step_outputs['acc'],
                'f1': step_outputs['f1'],
                'prec': step_outputs['prec'],
                'spec': step_outputs['spec'],
                'preds': step_outputs['preds'],
                'labels': step_outputs['labels'],
                'majority_preds': majority_preds,
                'majority_labels': majority_labels,
                'majority_prec': majority_prec,
                'majority_spec': majority_spec,
                'majority_acc': majority_acc,
                'majority_f1': majority_f1}

    def _connect_epoch_results(self, step_outputs: List[Dict[str, Tensor or Any]], key: str):
        to_concat = []
        for output in step_outputs:
            to_concat.append(output[key].detach().cpu())
        return torch.cat(to_concat)
