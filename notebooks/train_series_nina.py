from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, Specificity, Precision, F1Score

from datasets.stored_series import CapgMyoDataModule, MyoArmbandDataModule, NinaProDataModule
from definitions import MODELS_FOLDER
from models import Classifier
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import Compose, ToTensor, Normalize
import warnings
from functools import partial
from utils import cross_val_experiment, xgb_cross_val_experiments_file

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    seed = 113
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])
    callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                 partial(EarlyStopping, monitor='val/Accuracy', patience=7, mode='max')]

    data_module_ninapro = NinaProDataModule(
        batch_size=1000,
        k_folds=10,
        num_workers=32,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        seed=seed,
        series_length=2
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_ninapro.num_classes),
                                Specificity(average='macro', num_classes=data_module_ninapro.num_classes),
                                Precision(average='macro', num_classes=data_module_ninapro.num_classes),
                                F1Score(average='macro', num_classes=data_module_ninapro.num_classes)]).to(
        torch.device("cuda", 0))

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val/Accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=[19], time_step=[1],
                                 window_fix=[9], metrics=metrics)
    cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier, name="2 Series NinaPro",
                         max_epochs=150, callbacks=callbacks, seed=seed, model_checkpoint_index=0)


if __name__ == '__main__':
    main()
