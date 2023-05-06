from typing import Any, Dict, List

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import statistics
from torchmetrics import MetricCollection


class Classifier(pl.LightningModule):
    """
    Module performing all tasks of classification of given data with some model.
    It is responsible for training, validation, testing and prediction.
    It defines how those processes are organised.
    It measures all metrics and performs majority voting.
    """
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
        """
        Args:
            model: Model to be used for classification
            optimizer: optimizer for model training
            lr_scheduler: Type of learning rate scheduler, which will be used in training
            criterion: Criterion function for training
            time_window: List of Numbers of samples to perform majority voting of which
            time_step: List of numbers of samples that are ignored until next majority voting is performed. Their order coresponds to order in time_window
            window_fix: List of numbers that are equal to number of records in one sample minus one
            metrics: Collection of Metrics to be computed for classification results
            optim_kwargs: Dictionary of parameters for optimizer.
            sched_kwargs: Dictionary of parameters for scheduler.
            monitor: Name of metric to monitor for scheduler. Some schedulers are changing learning rate based on that metric.
            **kwargs
        """
        super().__init__()
        self.save_hyperparameters(ignore=['lr_lambda'])
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
        """
        Computing prediction for given matrix.
        Args:
            x: Tensor representing picture

        Returns:
            Vector of values representing probability of picture being each class
        """
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Dict[str, LambdaLR or None] or Any]:
        """
            Create optimizer and learning rate scheduler.
        Returns:
            Dictionary with optimizer and configured learning rate scheduler
        """
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
        """
            Perform a training step and log loss
        Args:
            train_batch: Batch of training data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(train_batch)
        logs = {'loss': results['loss']}
        logs = self._add_prefix_to_metrics('train/', logs)
        self.log_dict(logs)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """
            Finish training epoch and log all metrics results
        Args:
            outputs: Collected results of all steps
        """
        results = self._epoch_end(outputs)
        logs = self._add_prefix_to_metrics('train/', results['measurements'])
        self.log_dict(logs)

    def validation_step(self, val_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        """
            Perform a validation step
        Args:
            val_batch: Batch of validation data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(val_batch)
        return results

    def validation_epoch_end(self, validation_step_outputs: EPOCH_OUTPUT) -> None:
        """
            Finish validation epoch and log all metrics results and loss
        Args:
            validation_step_outputs: Collected results of all steps
        """
        results = self._epoch_end(validation_step_outputs)
        results = self._eval_epoch_end(results)['measurements']
        logs = self._add_prefix_to_metrics('val/', results)
        self.log_dict(logs)

    def test_step(self, test_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        """
            Perform a test step
        Args:
            test_batch: Batch of test data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(test_batch)
        return results

    def test_epoch_end(self, test_step_outputs: EPOCH_OUTPUT) -> None:
        """
            Finish test epoch and log all metrics results and loss
        Args:
            test_step_outputs: Collected results of all steps
        """
        results = self._epoch_end(test_step_outputs)
        results = self._eval_epoch_end(results)
        results = self._vote(results)['measurements']
        logs = self._add_prefix_to_metrics('test/', results)

        self.log_dict(logs)
        self.trainer.logger.finalize('success')

    def _add_prefix_to_metrics(self, prefix: str, logs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Add prefix to all keys in dictionary
        Args:
            prefix: Train, val or test
            logs: Dictionary with measured metrics

        Returns:
            Dictionary wit added prefixes
        """
        logs = {(prefix + key): value for key, value in logs.items()}
        return logs

    def _calculate_metrics(self, preds: Tensor, targets: Tensor) -> Dict[str, Tensor or List]:
        """
        Calculate all metrics
        Args:
            preds: Predictions.
            targets: Labels.

        Returns:
            Calculated metrics
        """
        metrics = self.metrics(preds, targets)
        return metrics

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        """
            Perform pure prediction of data
        Args:
            predict_batch: batch of data
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            Predicted classes
        """
        return self.model(predict_batch)

    def _step(self, batch: Dict[str, Tensor or Any]) -> Dict[str, Tensor or Any]:
        """
            Base of training, validation and test steps
        Args:
            batch: Batch of data

        Returns:
            Dictionary with: loss, predictions, labels, series numbers and indexes
        """
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
        """
            Compute a moving average over dataframe
        Args:
            df: Dataframe with 2 columns: preds and labels
            window: Amount of values to compute moving average on.
            step: Step of moving average.

        Returns:
            Computed moving average of 2 columns: preds and labels
        """
        if df.shape[0] >= window:
            preds = []
            labels = []
            for i in range((df.shape[0] - window) // step + 1):
                tmp = df.iloc[(i * step):(i * step + window)]
                preds.append(tmp['preds'].values.tolist())
                labels.append(tmp['labels'].values.tolist())
            return {'preds_list': preds, 'labels_list': labels}
        else:
            return {'preds': df['preds'].mode()[0].item(), 'labels': df['labels'].mode()[0].item()}

    def _majority_voting(self, df: pd.DataFrame, window: int, step: int) -> STEP_OUTPUT:
        """
            Perform a majority voting over dataframe and compute metrics for the results.
        Args:
            df: Dataframe with 3 columns: preds, labels and spectrograms
            window: Amount of values to compute moving average on.
            step: Step of moving average.

        Returns:
            Dictionary with computed metrics for voted majorities
        """
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
        labels += torch.mode(torch.Tensor(labels_list))[0].int().tolist()

        majority_preds = torch.tensor(preds, device=self.device)
        majority_labels = torch.tensor(labels, device=self.device)

        return self._add_prefix_to_metrics('majority_voting_', self._calculate_metrics(majority_preds, majority_labels))

    def _epoch_end(self, step_outputs: EPOCH_OUTPUT) -> STEP_OUTPUT:
        """
            End an epoch and log all the computed metrics.
        Args:
            step_outputs: Collected results of all steps

        Returns:
            Dictionary containing computed measurements and outputs from model
        """
        preds = self._connect_epoch_results(step_outputs, 'preds')
        labels = self._connect_epoch_results(step_outputs, 'labels')
        series = self._connect_epoch_results(step_outputs, 'spectrograms')
        index = self._connect_epoch_results(step_outputs, 'index')
        loss = [step['loss'] for step in step_outputs]
        output = {'preds': preds,
                  'labels': labels,
                  'spectrograms': series,
                  'index': index,
                  'loss': loss}
        measurements = self._calculate_metrics(preds.to(self.device), labels.to(self.device))
        return {'output': output, 'measurements': measurements}

    def _eval_epoch_end(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """
            End validation or test epoch and log all the computed metrics.
        Args:
            step_outputs: Collected results of all steps

        Returns:
            Dictionary containing computed measurements and outputs from model
        """
        output = step_outputs['output']
        measurements = step_outputs['measurements']
        measurements.update({'loss': statistics.fmean(output['loss'])})
        output.pop('loss', None)
        return {'output': output, 'measurements': measurements}

    def _vote(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """
            Start series of majority voting for each size of voting window defined in classifier.
        Args:
            step_outputs:

        Returns: Dictionary containing computed measurements and outputs from model

        """
        output = step_outputs['output']
        measurements = step_outputs['measurements']
        df = pd.DataFrame(output).sort_values(by=['index'])
        if self.window_fix is None:
            for window, step in zip(self.time_window, self.time_step):
                results = self._majority_voting(df, window, step)
                measurements.update(self._add_prefix_to_metrics(f'{window}_{step}/', results))
        else:
            for window, step, fix in zip(self.time_window, self.time_step, self.window_fix):
                results = self._majority_voting(df, window, step)
                measurements.update(self._add_prefix_to_metrics(f'{window + fix}_{step}/', results))
        return {'output': output, 'measurements': measurements}

    def _connect_epoch_results(self, step_outputs: List[Dict[str, Tensor or Any]], key: str) -> Tensor:
        """
            Connect values from dictionaries from all steps into one for the whole epoch
        Args:
            step_outputs: List of dictionaries to connect
            key: key to dictionary to specify, which values will be connected

        Returns:
            Connected values
        """
        to_concat = []
        for output in step_outputs:
            to_concat.append(output[key].detach().cpu())
        return torch.cat(to_concat)
