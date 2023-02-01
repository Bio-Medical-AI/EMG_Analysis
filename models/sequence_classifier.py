from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from models import CRNN
from torch.nn.utils.rnn import pack_padded_sequence

from models import Classifier


class SequenceClassifier(Classifier):
    def __init__(self,
                 model: CRNN,
                 optimizer: type(torch.optim.Optimizer) = torch.optim.AdamW,
                 lr_scheduler: type(torch.optim.lr_scheduler) = torch.optim.lr_scheduler.ReduceLROnPlateau,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 time_window: List[int] = [],
                 time_step: List[int] = [],
                 window_fix: List[int] = None,
                 metrics: MetricCollection = MetricCollection([]),
                 **kwargs):
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
