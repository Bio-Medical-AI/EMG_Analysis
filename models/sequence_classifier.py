from typing import Any, Dict, List

import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import MetricCollection
from models import RNN
from torch.nn.utils.rnn import pack_padded_sequence

from models import Classifier


class SequenceClassifier(Classifier):
    """
    Module performing all tasks of classification of given data with some recurrent model.
    It is responsible for training, validation, testing and prediction.
    It defines how those processes are organised.
    It measures all metrics and performs majority voting.
    """
    def __init__(self,
                 model: RNN,
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
        super(SequenceClassifier, self).__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            time_window=time_window,
            time_step=time_step,
            window_fix=window_fix,
            metrics=metrics,
            **kwargs)

    def _step(self, batch: Dict[str, Tensor or Any]) -> Dict[str, Tensor or Any]:
        batch['label'] = torch.cat(
            [labels[:length] for labels, length in zip(batch['label'], batch['length'])])
        batch['spectrograms'] = torch.cat(
            [torch.ones(length, device=self.device, dtype=torch.long) * spec
             for spec, length in zip(batch['spectrograms'], batch['length'])])
        batch['index'] = torch.cat(
            [torch.arange(length, device=self.device, dtype=torch.long) +
             batch['data'].shape[2] * idx for idx, length in zip(batch['index'], batch['length'])])
        batch['data'] = pack_padded_sequence(torch.squeeze(batch['data']), batch['length'].cpu(), batch_first=True,
                                             enforce_sorted=False)
        return super()._step(batch)
