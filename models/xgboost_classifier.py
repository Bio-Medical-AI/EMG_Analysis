from torch.optim import AdamW
from xgboost import XGBClassifier

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn as nn
from torchmetrics.functional import accuracy, f1_score, precision, specificity


class LightningXGBClassifier(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 num_class: int,
                 objective: str = 'multi:softprob',
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 tree_method: str = 'hist'):
        super().__init__()
        self.model = model
        self.xgbmodel = XGBClassifier(objective=objective, num_class=num_class, tree_method=tree_method, gpu_id=0)
        self.criterion = criterion

    def forward(self, x: Tensor) -> Tensor:
        return torch.from_numpy(self.xgbmodel.predict_proba(self.model(x).detach().cpu().numpy())).requires_grad_()

    def _fit(self, x: Tensor, y: Tensor):
        x_fit = self.model(x).detach().cpu().numpy()
        y_fit = y.detach().cpu().numpy()
        return self.xgbmodel.fit(x_fit, y_fit)

    def configure_optimizers(self) -> AdamW:
        return torch.optim.AdamW(self.model.parameters())

    def training_step(self, train_batch: tuple[Tensor, Tensor], batch_idx: int) -> STEP_OUTPUT:
        x, y = train_batch
        self._fit(x, y)

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
