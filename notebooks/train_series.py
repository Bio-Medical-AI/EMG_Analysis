from datasets.stored_series import CapgMyoDataModule, MyoArmbandDataModule, NinaProDataModule
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
    lr_lambda = lambda epoch: 1 if epoch < 16 else 0.1 if epoch < 24 else 0.01
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=7, mode='max')]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR
    optimizer = torch.optim.AdamW

    data_module_capgmyo = CapgMyoDataModule(
        batch_size=1000,
        k_folds=10,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        seed=seed,
        series_length=10
    )

    partial_classifier = partial(Classifier, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                 optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 lr_lambda=lr_lambda, time_window=30, time_step=1)
    cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="Series Chinese CapgMyo",
                         max_epochs=28, seed=seed)

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=30, time_step=1)
    cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="Series CapgMyo",
                         max_epochs=150, callbacks=callbacks, seed=seed)

    partial_classifier = partial(Classifier, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                 optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 lr_lambda=lr_lambda, time_window=140, time_step=1)
    cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="Series Chinese CapgMyo 150",
                         max_epochs=28, seed=seed)

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=140, time_step=1)
    cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="Series CapgMyo 150",
                         max_epochs=150, callbacks=callbacks, seed=seed)

    data_module_myoarmband = MyoArmbandDataModule(
        batch_size=1000,
        num_workers=8,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        k_folds=6,
        seed=seed,
        series_length=1
    )
    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=30, time_step=1)
    cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier, name="Series MyoArmband",
                         max_epochs=150, callbacks=callbacks, seed=seed)

    data_module_ninapro = NinaProDataModule(
        batch_size=100,
        k_folds=10,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        seed=seed,
        series_length=1
    )

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val_accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=18, time_step=1)
    cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier, name="Series NinaPro",
                         max_epochs=150, callbacks=callbacks, seed=seed)


if __name__ == '__main__':
    main()
