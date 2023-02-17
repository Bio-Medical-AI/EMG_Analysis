from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, Specificity, Precision, F1Score

from datasets import SpectrogramDataset
from datasets.frames import CapgMyoDataModule, MyoArmbandDataModule, NinaProDataModule, KNIBMHighDataModule, \
    KNIBMLowDataModule
from definitions import MODELS_FOLDER
from models import SequenceClassifier, CRNN, Classifier, OriginalModel
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import Compose, ToTensor, Normalize
import warnings
from functools import partial
from utils import lstm_cross_val_experiment

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    max_epochs = 150
    max_seq_epochs = 150
    seed = 113
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])
    sequence_model = partial(CRNN, num_layers=2, dropout=0.5, hidden_size=128)

    callbacks = [partial(ModelCheckpoint, monitor='val/F1Score', dirpath=MODELS_FOLDER, mode='max'),
                 partial(EarlyStopping, monitor='val/F1Score', patience=11, mode='max')]

    seq_callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                     partial(EarlyStopping, monitor='val/loss', patience=17)]
    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.0001, 'weight_decay': 0.001},
                                 monitor='val/F1Score', sched_kwargs={'patience': 7, 'mode': 'max'},
                                 time_window=[10, 20, 40], time_step=[1, 1, 1])
    partial_seq_classifier = partial(SequenceClassifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.001},
                                     monitor='val/loss', sched_kwargs={'patience': 13})

    data_module_knibm_low = KNIBMLowDataModule(
        batch_size=10000,
        k_folds=5,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        num_workers=32,
        seed=seed,
        shuffle_train=True,
        dataset=SpectrogramDataset,
        window_length=1
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_knibm_low.num_classes),
                                Specificity(average='macro', num_classes=data_module_knibm_low.num_classes),
                                Precision(average='macro', num_classes=data_module_knibm_low.num_classes),
                                F1Score(average='macro', num_classes=data_module_knibm_low.num_classes)]).to(
        torch.device("cuda", 0))

    lstm_cross_val_experiment(data_module=data_module_knibm_low, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM low {data_module_knibm_low.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_low.set_window_length(2)
    lstm_cross_val_experiment(data_module=data_module_knibm_low, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM low {data_module_knibm_low.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_low.set_window_length(5)
    lstm_cross_val_experiment(data_module=data_module_knibm_low, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM low {data_module_knibm_low.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_low.set_window_length(10)
    lstm_cross_val_experiment(data_module=data_module_knibm_low, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM low {data_module_knibm_low.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_low = None


    data_module_knibm_high = KNIBMHighDataModule(
        batch_size=10000,
        k_folds=5,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        num_workers=32,
        seed=seed,
        shuffle_train=True,
        dataset=SpectrogramDataset,
        window_length=1
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_knibm_high.num_classes),
                                Specificity(average='macro', num_classes=data_module_knibm_high.num_classes),
                                Precision(average='macro', num_classes=data_module_knibm_high.num_classes),
                                F1Score(average='macro', num_classes=data_module_knibm_high.num_classes)]).to(
        torch.device("cuda", 0))

    lstm_cross_val_experiment(data_module=data_module_knibm_high, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM high {data_module_knibm_high.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_high.set_window_length(2)
    lstm_cross_val_experiment(data_module=data_module_knibm_high, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM high {data_module_knibm_high.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_high.set_window_length(5)
    lstm_cross_val_experiment(data_module=data_module_knibm_high, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM high {data_module_knibm_high.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_high.set_window_length(10)
    lstm_cross_val_experiment(data_module=data_module_knibm_high, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'KNIBM high {data_module_knibm_high.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_knibm_high = None

    callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                 partial(EarlyStopping, monitor='val/loss', patience=7)]

    seq_callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                     partial(EarlyStopping, monitor='val/loss', patience=17)]

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001},
                                 monitor='val/loss', sched_kwargs={'patience': 4},
                                 time_window=[10, 20, 40], time_step=[1, 1, 1])
    partial_seq_classifier = partial(SequenceClassifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.001},
                                     monitor='val/loss', sched_kwargs={'patience': 13})


    data_module_capgmyo = CapgMyoDataModule(
        batch_size=10000,
        k_folds=10,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        num_workers=32,
        seed=seed,
        shuffle_train=True,
        dataset=SpectrogramDataset,
        window_length=1
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_capgmyo.num_classes),
                                Specificity(average='macro', num_classes=data_module_capgmyo.num_classes),
                                Precision(average='macro', num_classes=data_module_capgmyo.num_classes),
                                F1Score(average='macro', num_classes=data_module_capgmyo.num_classes)]).to(
        torch.device("cuda", 0))

    lstm_cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'CapgMyo {data_module_capgmyo.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_capgmyo.set_window_length(2)
    lstm_cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'CapgMyo {data_module_capgmyo.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_capgmyo.set_window_length(5)
    lstm_cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'CapgMyo {data_module_capgmyo.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_capgmyo.set_window_length(10)
    lstm_cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier,
                              name=f'CapgMyo {data_module_capgmyo.window_length}',
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
                              sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)

    data_module_capgmyo = None




    # data_module_myoarmband = MyoArmbandDataModule(
    #     batch_size=10000,
    #     k_folds=6,
    #     train_transforms=transform,
    #     val_transforms=transform,
    #     test_transforms=transform,
    #     num_workers=32,
    #     seed=seed,
    #     shuffle_train=True,
    #     dataset=SpectrogramDataset,
    #     window_length=1
    # )
    #
    # metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_myoarmband.num_classes),
    #                             Specificity(average='macro', num_classes=data_module_myoarmband.num_classes),
    #                             Precision(average='macro', num_classes=data_module_myoarmband.num_classes),
    #                             F1Score(average='macro', num_classes=data_module_myoarmband.num_classes)]).to(
    #     torch.device("cuda", 0))
    #
    # lstm_cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'MyoArmband {data_module_myoarmband.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_myoarmband.set_window_length(2)
    # lstm_cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'MyoArmband {data_module_myoarmband.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_myoarmband.set_window_length(5)
    # lstm_cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'MyoArmband {data_module_myoarmband.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_myoarmband.set_window_length(10)
    # lstm_cross_val_experiment(data_module=data_module_myoarmband, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'MyoArmband {data_module_myoarmband.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_myoarmband = None

    # data_module_ninapro = NinaProDataModule(
    #     batch_size=10000,
    #     num_workers=24,
    #     k_folds=10,
    #     train_transforms=transform,
    #     val_transforms=transform,
    #     test_transforms=transform,
    #     seed=seed,
    #     shuffle_train=True,
    #     dataset=SpectrogramDataset,
    #     window_length=1
    # )
    #
    # metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_ninapro.num_classes),
    #                             Specificity(average='macro', num_classes=data_module_ninapro.num_classes),
    #                             Precision(average='macro', num_classes=data_module_ninapro.num_classes),
    #                             F1Score(average='macro', num_classes=data_module_ninapro.num_classes)]).to(
    #     torch.device("cuda", 0))
    #
    # lstm_cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'NinaPro {data_module_ninapro.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model, seq_batch_size=30,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_ninapro.set_window_length(2)
    # lstm_cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'NinaPro {data_module_ninapro.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model, seq_batch_size=30,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_ninapro.set_window_length(5)
    # lstm_cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'NinaPro {data_module_ninapro.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model, seq_batch_size=30,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)
    #
    # data_module_ninapro.set_window_length(10)
    # lstm_cross_val_experiment(data_module=data_module_ninapro, partial_classifier=partial_classifier,
    #                           partial_seq_classifier=partial_seq_classifier,
    #                           name=f'NinaPro {data_module_ninapro.window_length}',
    #                           max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, metrics=metrics,
    #                           sequence_model=sequence_model, seq_batch_size=30,
    #                           callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)


if __name__ == "__main__":
    main()
