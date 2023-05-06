from typing import List

from torch import nn
from torchmetrics import MetricCollection

from datasets import AbstractDataModule, SequenceDataModule
from models import OriginalModel, UniLSTM
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from models import LightningXGBClassifier
import os
from definitions import MODELS_FOLDER
from functools import partial
from tqdm import tqdm
import wandb
from definitions import WANDB_PROJECT_NAME


def cross_val_experiment(data_module: AbstractDataModule, partial_classifier: partial, name: str, max_epochs: int,
                         callbacks: list = None, model_checkpoint_index: int = None, project: str = WANDB_PROJECT_NAME,
                         save_dir: str = 'wandb_logs', seed: int = 0, classifier_params: dict = {},
                         model_type: type(nn.Module) = OriginalModel) -> None:
    """
    Perform base experiment with cross-validation
    Args:
        data_module: Data module representing dataset.
        partial_classifier: Partially initialized classifier, that doesn't have model defined yet.
        name: Display name for the run.
        max_epochs: Maximum amount of epochs to perform. To enable infinite training, set max_epochs to -1.
        callbacks: List of partially initialized callbacks to use in training. If model_checkpoint_index isn't None, then there must be early stopping callback in that list.
        model_checkpoint_index: index of early stopping callback in list of callbacks
        project: Name of wandb project
        save_dir: Path to directory in which logs will be saved
        seed: Seed for randomization
        classifier_params: params, that will be passed to model
        model_type: Type of model used by classifier
    """
    pl.seed_everything(seed, workers=True)
    if not classifier_params:
        classifier_params = data_module.get_data_parameters()
    k_folds = data_module.k_folds
    for k in tqdm(range(k_folds)):
        classifier = partial_classifier(
            model_type(**classifier_params))
        logger = WandbLogger(project=project, name=name, save_dir=save_dir)
        if callbacks:
            callbacks_initialized = [callback() for callback in callbacks]
        else:
            callbacks_initialized = None
        trainer = pl.Trainer(gpus=-1, max_epochs=max_epochs, logger=logger, accelerator="gpu",
                             callbacks=callbacks_initialized)
        trainer.fit(model=classifier, datamodule=data_module)
        if model_checkpoint_index is not None and bool(callbacks):
            classifier = classifier.load_from_checkpoint(
                checkpoint_path=callbacks_initialized[model_checkpoint_index].best_model_path)
        trainer.test(model=classifier, datamodule=data_module)
        wandb.finish()
        torch.save(classifier.model.state_dict(),
                   os.path.join(MODELS_FOLDER, f'{name.replace(" ", "_").lower()}_{k}_fold.pt'))
        data_module.next_fold()


def lstm_cross_val_experiment(data_module: AbstractDataModule, partial_classifier: partial,
                              partial_seq_classifier: partial, name: str, max_epochs: int, max_seq_epochs: int,
                              metrics: MetricCollection, sequence_model: partial = partial(UniLSTM),
                              callbacks: list = None, seq_callbacks: list = None, model_checkpoint_index: int = None,
                              project: str = WANDB_PROJECT_NAME, save_dir: str = 'wandb_logs', seed: int = 0,
                              classifier_params: dict = {}, model_type: type(nn.Module) = OriginalModel,
                              seq_batch_size: int = 12):
    """
    Perform experiment with cross-validation for CRNN with separate convolution training and lstm training
    Args:
        data_module: Data module representing dataset.
        partial_classifier: Partially initialized classifier, that doesn't have model defined yet. Used for non-sequential classification
        partial_seq_classifier: Partially initialized classifier, that doesn't have model defined yet. Used for sequential classification.
        name: Display name for the run.
        max_epochs: Maximum amount of epochs to perform by non-sequential classifier. To enable infinite training, set max_epochs to -1.
        max_seq_epochs: Maximum amount of epochs to perform by sequential classifier. To enable infinite training, set max_epochs to -1.
        metrics: collection of metrics used for evaluation of model result
        sequence_model: Partially initialized model for sequence classification
        callbacks: List of partially initialized callbacks to use in training non-sequential model. If model_checkpoint_index isn't None, then there must be early stopping callback in that list.
        seq_callbacks: List of partially initialized callbacks to use in training sequential model. If model_checkpoint_index isn't None, then there must be early stopping callback in that list.
        model_checkpoint_index: index of early stopping callback in list of callbacks
        project: Name of wandb project
        save_dir: Path to directory in which logs will be saved
        seed: Seed for randomization
        classifier_params: params, that will be passed to model
        model_type: Type of model used by classifier
        seq_batch_size: batch size of sequential data
    """
    pl.seed_everything(seed, workers=True)
    if not classifier_params:
        classifier_params = data_module.get_data_parameters()
    k_folds = data_module.k_folds
    for k in tqdm(range(k_folds)):
        classifier = partial_classifier(
            model_type(**classifier_params), metrics=metrics)
        logger = WandbLogger(project=project, name=name, save_dir=save_dir)
        if callbacks:
            callbacks_initialized = [callback() for callback in callbacks]
        else:
            callbacks_initialized = None
        trainer = pl.Trainer(gpus=-1, max_epochs=max_epochs, logger=logger, accelerator="gpu",
                             callbacks=callbacks_initialized)
        trainer.fit(model=classifier, datamodule=data_module)
        if model_checkpoint_index is not None and bool(callbacks):
            classifier = classifier.load_from_checkpoint(
                checkpoint_path=callbacks_initialized[model_checkpoint_index].best_model_path)
        trainer.test(model=classifier, datamodule=data_module)
        wandb.finish()
        torch.save(classifier.model.state_dict(),
                   os.path.join(MODELS_FOLDER, f'{name.replace(" ", "_").lower()}_{k}_fold.pt'))

        seq_model = classifier.model.model[0:-1]
        seq_model.eval()
        seq_data_module = SequenceDataModule(df_path=data_module.df_path,
                                             width=data_module.width,
                                             height=data_module.height,
                                             channels=data_module.channels,
                                             num_classes=data_module.num_classes,
                                             feature_extraction=seq_model,
                                             splits_series=data_module.get_splits_series(),
                                             feature_extraction_dataset=
                                             partial(data_module.dataset, window_length=data_module.window_length),
                                             feature_extraction_transforms=data_module.test_transforms,
                                             window_length=100000,
                                             batch_size=seq_batch_size)

        seq_calssifier = partial_seq_classifier(
            model=sequence_model(
                input_size=classifier.model.model[-1].in_features,
                num_classes=classifier.model.model[-1].out_features),
            time_step=classifier.time_step,
            time_window=classifier.time_window,
            metrics=classifier.metrics,
        )

        logger = WandbLogger(project=project, name='lstm ' + name, save_dir=save_dir)
        if seq_callbacks:
            callbacks_initialized = [callback() for callback in seq_callbacks]
        else:
            callbacks_initialized = None
        trainer = pl.Trainer(gpus=-1, max_epochs=max_seq_epochs, logger=logger, accelerator="gpu",
                             callbacks=callbacks_initialized)
        trainer.fit(model=seq_calssifier, datamodule=seq_data_module)
        if model_checkpoint_index is not None and bool(callbacks):
            seq_calssifier = seq_calssifier.load_from_checkpoint(
                checkpoint_path=callbacks_initialized[model_checkpoint_index].best_model_path)
        trainer.test(model=seq_calssifier, datamodule=seq_data_module)
        wandb.finish()
        data_module.next_fold()
