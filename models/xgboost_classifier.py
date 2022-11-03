from typing import Any, Dict, List

from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from xgboost import XGBClassifier

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from models.classifier import Classifier


class LightningXGBClassifier(Classifier):
    def __init__(self,
                 model: nn.Module,
                 num_class: int,
                 objective: str = 'multi:softprob',
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 tree_method: str = 'hist',
                 time_window: List[int] = [30],
                 time_step: List[int] = [1],
                 metrics: MetricCollection = MetricCollection([]),
                 n_jobs: int = 16,
                 **kwargs
                 ):
        super(LightningXGBClassifier, self).__init__(
            model=model,
            time_window=time_window,
            time_step=time_step,
            metrics=metrics,
            **kwargs
        )
        self.xgbmodel = XGBClassifier(objective=objective, num_class=num_class, tree_method=tree_method, n_jobs=n_jobs,
                                      gpu_id=0)
        self.criterion = criterion

    def forward(self, x: Tensor) -> Tensor:
        return torch.from_numpy(self.xgbmodel.predict_proba(self.model(x).detach().cpu().numpy())).requires_grad_()

    def _fit(self, x: Tensor, y: Tensor):
        x_fit = self.model(x).detach().cpu().numpy()
        y_fit = y.detach().cpu().numpy()
        return self.xgbmodel.fit(x_fit, y_fit)

    def training_step(self, train_batch: Dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        self._fit(train_batch['data'], train_batch['label'])
        return super(LightningXGBClassifier, self).training_step(train_batch, batch_idx)

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        return self.model(predict_batch)
