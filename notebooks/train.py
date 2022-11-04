from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, Specificity, Precision, F1Score
from datasets.stored_dataframe import CapgMyoDataModule, MyoArmbandDataModule, NinaProDataModule
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
    lr_lambda = lambda epoch: 1 if epoch < 16 else 0.1 if epoch < 24 else 0.01
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])
    callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                 partial(EarlyStopping, monitor='val/Accuracy', patience=7, mode='max')]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR
    optimizer = torch.optim.AdamW

    # data_module_capgmyo = CapgMyoDataModule(
    #     batch_size=1000,
    #     k_folds=10,
    #     train_transforms=transform,
    #     val_transforms=transform,
    #     test_transforms=transform,
    #     seed=seed
    # )
    #
    # metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_capgmyo.num_classes),
    #                             Specificity(average='macro', num_classes=data_module_capgmyo.num_classes),
    #                             Precision(average='macro', num_classes=data_module_capgmyo.num_classes),
    #                             F1Score(average='macro', num_classes=data_module_capgmyo.num_classes)]).to(
    #     torch.device("cuda", 0))
    #
    # partial_classifier = partial(Classifier, optimizer=optimizer, lr_scheduler=lr_scheduler,
    #                              optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val/Accuracy',
    #                              lr_lambda=lr_lambda, time_window=[40], time_step=[1], metrics=metrics)
    # cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="Chinese CapgMyo",
    #                      max_epochs=28, seed=seed, model_checkpoint_index=None)
    #
    # partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val/Accuracy',
    #                              sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=[40], time_step=[1],
    #                              metrics=metrics)
    # cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier, name="CapgMyo",
    #                      max_epochs=150, callbacks=callbacks, seed=seed, model_checkpoint_index=0)

    # model_files = [f'capgmyo_{k}_fold.pt' for k in range(10)]
    # xgb_cross_val_experiments_file(data_module=data_module_capgmyo, model_files=model_files, name="XGB CapgMyo",
    #                                max_epochs=10, seed=seed, time_window=[40], time_step=[1], metrics=metrics)


    # data_module_myoarmband = MyoArmbandDataModule(
    #     batch_size=10000,
    #     num_workers=8,
    #     train_transforms=transform,
    #     val_transforms=transform,
    #     test_transforms=transform,
    #     k_folds=6,
    #     seed=seed
    # )
    #
    # metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_myoarmband.num_classes),
    #                             Specificity(average='macro', num_classes=data_module_myoarmband.num_classes),
    #                             Precision(average='macro', num_classes=data_module_myoarmband.num_classes),
    #                             F1Score(average='macro', num_classes=data_module_myoarmband.num_classes)]).to(
    #         torch.device("cuda", 0))

    # partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val/Accuracy',
    #                              sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=[40], time_step=[1],
    #                              metrics=metrics)
    # cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier, name="MyoArmband",
    #                      max_epochs=150, callbacks=callbacks, seed=seed, k_folds=6, model_checkpoint_index=0)

    # model_files = [f'myoarmband_{k}_fold.pt' for k in range(10)]
    # xgb_cross_val_experiments_file(data_module=data_module_myoarmband, model_files=model_files, name="XGB MyoArmband",
    #                                max_epochs=10, seed=seed, time_window=[40], time_step=[1], metrics=metrics)

    data_module_ninapro = NinaProDataModule(
        batch_size=5000,
        num_workers=32,
        k_folds=10,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        seed=seed
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_ninapro.num_classes),
                                Specificity(average='macro', num_classes=data_module_ninapro.num_classes),
                                Precision(average='macro', num_classes=data_module_ninapro.num_classes),
                                F1Score(average='macro', num_classes=data_module_ninapro.num_classes)]).to(
            torch.device("cuda", 0))

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001}, monitor='val/Accuracy',
                                 sched_kwargs={'patience': 4, 'mode': 'max'}, time_window=[28], time_step=[1],
                                 metrics=metrics)
    cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier, name="NinaPro",
                         max_epochs=150, callbacks=callbacks, seed=seed, model_checkpoint_index=0)

    # model_files = [f'ninapro_{k}_fold.pt' for k in range(10)]
    # xgb_cross_val_experiments_file(data_module=data_module_ninapro, model_files=model_files, name="XGB NinaPro",
    #                                max_epochs=10, seed=seed, time_window=[28], time_step=[1], metrics=metrics)


if __name__ == '__main__':
    main()
