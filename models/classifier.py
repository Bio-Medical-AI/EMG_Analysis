from typing import Any, Dict, List

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import statistics
from torchmetrics import MetricCollection


class Classifier(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.AdamW,
                 lr_scheduler: type(torch.optim.lr_scheduler) = torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 time_window: List[int] = [],
                 time_step: List[int] = [],
                 window_fix: List[int] = None,
                 metrics: MetricCollection = MetricCollection([]),
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['lr_lambda'])
        #                                   'criterion',
        #                                   'lr_lambda',
        #                                   'lr_scheduler',
        #                                   'optimizer',
        #                                   'metrics',
        #                                   'time_window',
        #                                   'time_step'])
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.time_window = time_window
        self.time_step = time_step
        self.window_fix = window_fix
        self.metrics = metrics
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
        logs = self._add_prefix_to_metrics('train/', logs)
        self.log_dict(logs)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        results = self._epoch_end(outputs)
        logs = self._add_prefix_to_metrics('train/', results['measurements'])
        self.log_dict(logs)

    def validation_step(self, val_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(val_batch)
        return results

    def validation_epoch_end(self, validation_step_outputs: EPOCH_OUTPUT) -> None:
        results = self._epoch_end(validation_step_outputs)
        results = self._eval_epoch_end(results)
        logs = self._add_prefix_to_metrics('val/', results)
        self.log_dict(logs)

    def test_step(self, test_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        results = self._step(test_batch)
        return results

    def test_epoch_end(self, test_step_outputs: EPOCH_OUTPUT) -> None:
        results = self._epoch_end(test_step_outputs)
        results = self._eval_epoch_end(results)
        logs = self._add_prefix_to_metrics('test/', results)

        self.log_dict(logs)
        self.trainer.logger.finalize('success')

    def _add_prefix_to_metrics(self, prefix: str, logs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        logs = {(prefix + key): value for key, value in logs.items()}
        return logs

    def _calculate_metrics(self, preds: Tensor, targets: Tensor) -> Dict[str, Tensor or List]:
        metrics = self.metrics(preds, targets)
        return metrics

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)

    def _step(self, batch: Dict[str, Tensor or Any]) -> Dict[str, Tensor or Any]:
        x = batch['data']
        y = batch['label']
        series = batch['spectrograms']
        index = batch['index']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss,
                'preds': preds,
                'labels': y,
                'spectrograms': series,
                'index': index}

    def _moving_average(self, df: pd.DataFrame, window: int, step: int) -> STEP_OUTPUT:
        if df.shape[0] >= window:
            preds = []
            labels = []
            for i in range((df.shape[0] - window) // step + 1):
                tmp = df.iloc[(i * step):(i * step + window)]
                preds.append(tmp['preds'].values.tolist())
                labels.append(tmp['labels'].head(1).item())
            return {'preds_list': preds, 'labels_list': labels}
        else:
            return {'preds': df['preds'].mode()[0].item(), 'labels': df['labels'].head(1).item()}

    def _majority_voting(self, df: pd.DataFrame, window: int, step: int) -> STEP_OUTPUT:
        preds = []
        labels = []
        preds_list = []
        labels_list = []

        for series in df['spectrograms'].unique().tolist():
            results = self._moving_average(df.loc[df['spectrograms'] == series], window, step)
            if 'preds_list' in results.keys():
                preds_list += results['preds_list']
                labels_list += results['labels_list']
            else:
                preds.append(results['preds'])
                labels.append(results['labels'])
        preds += torch.mode(torch.Tensor(preds_list))[0].int().tolist()
        labels += labels_list

        majority_preds = torch.tensor(preds, device=self.device)
        majority_labels = torch.tensor(labels, device=self.device)

        # majority_acc = accuracy(majority_preds, majority_labels, average='micro')
        # majority_f1 = f1_score(majority_preds, majority_labels, average='macro', num_classes=self.model.num_classes)
        # majority_prec = precision(majority_preds, majority_labels, average='macro', num_classes=self.model.num_classes)
        # majority_spec = specificity(majority_preds, majority_labels, average='macro',
        #                             num_classes=self.model.num_classes)
        return self._add_prefix_to_metrics('majority_voting_', self._calculate_metrics(majority_preds, majority_labels))

    def _epoch_end(self, step_outputs: EPOCH_OUTPUT) -> STEP_OUTPUT:
        preds = self._connect_epoch_results(step_outputs, 'preds')
        labels = self._connect_epoch_results(step_outputs, 'labels')
        series = self._connect_epoch_results(step_outputs, 'spectrograms')
        index = self._connect_epoch_results(step_outputs, 'index')
        loss = [step['loss'] for step in step_outputs]
        # acc = accuracy(preds, labels, average='micro')
        # f1 = f1_score(preds, labels, average='macro', num_classes=self.model.num_classes)
        # prec = precision(preds, labels, average='macro', num_classes=self.model.num_classes)
        # spec = specificity(preds, labels, average='macro', num_classes=self.model.num_classes)
        output = {'preds': preds,
                  'labels': labels,
                  'spectrograms': series,
                  'index': index,
                  'loss': loss}
        measurements = self._calculate_metrics(preds.to(self.device), labels.to(self.device))
        return {'output': output, 'measurements': measurements}

    def _eval_epoch_end(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        output = step_outputs['output']
        measurements = step_outputs['measurements']
        measurements.update({'loss': statistics.fmean(output['loss'])})
        output.pop('loss', None)
        df = pd.DataFrame(output).sort_values(by=['index'])
        if self.window_fix is None:
            for window, step in zip(self.time_window, self.time_step):
                results = self._majority_voting(df, window, step)
                measurements.update(self._add_prefix_to_metrics(f'{window}_{step}/', results))
        else:
            for window, step, fix in zip(self.time_window, self.time_step, self.window_fix):
                results = self._majority_voting(df, window, step)
                measurements.update(self._add_prefix_to_metrics(f'{window + fix}_{step}/', results))

        return measurements

    def _connect_epoch_results(self, step_outputs: List[Dict[str, Tensor or Any]], key: str):
        to_concat = []
        for output in step_outputs:
            to_concat.append(output[key].detach().cpu())
        return torch.cat(to_concat)
