from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from torchmetrics import MetricCollection
from models import CRNN

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
        self.batch_size = self.model.sequence_length

    def _step(self, batch: Dict[str, Tensor or Any]) -> Dict[str, Tensor or Any]:
        batch['data'] = batch['data'][0]
        if len(batch['data'].shape) == 3:
            batch['data'] = batch['data'][None].permute((1, 0, 2, 3)).contiguous()
        elif len(batch['data'].shape) == 2:
            batch['data'] = batch['data'][None, None, :, :].permute((2, 1, 0, 3)).contiguous()
        elif len(batch['data'].shape) == 1:
            batch['data'] = batch['data'][None, None, None, :].permute((3, 1, 2, 0)).contiguous()
        batch['label'] = torch.ones(batch['data'].shape[0], device=self.device, dtype=torch.long) * batch['label'][0]
        batch['spectrograms'] = \
            torch.ones(batch['data'].shape[0], device=self.device, dtype=torch.long) * batch['spectrograms'][0]
        batch['index'] = torch.from_numpy(np.arange(batch['data'].shape[0])).to(self.device) + \
                         self.batch_size * batch['index'][0]
        return super()._step(batch)
