from typing import List

from datasets import AbstractDataModule
from models import OriginalModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from models import LightningXGBClassifier
import os
from definitions import MODELS_FOLDER
from functools import partial
from tqdm import tqdm
import wandb
import copy


def cross_val_experiment(data_module: AbstractDataModule, partial_classifier: partial, name: str, max_epochs: int,
                         callbacks: list = None, project: str = "EMG Armband", save_dir: str = 'wandb_logs',
                         k_folds: int = 10, seed: int = 0):
    pl.seed_everything(seed, workers=True)
    parameters = data_module.get_data_parameters()
    for k in tqdm(range(k_folds)):
        classifier = partial_classifier(
            OriginalModel(**parameters))
        logger = WandbLogger(project=project, name=name, save_dir=save_dir)
        trainer = pl.Trainer(gpus=-1, max_epochs=max_epochs, logger=logger, accelerator="gpu",
                             callbacks=copy.deepcopy(callbacks))
        trainer.fit(model=classifier, datamodule=data_module)
        trainer.test(model=classifier, datamodule=data_module)
        wandb.finish()
        torch.save(classifier.model.state_dict(),
                   os.path.join(MODELS_FOLDER, f'{name.replace(" ", "_").lower()}_{k}_fold.pt'))
        data_module.next_fold()


def xgb_cross_val_experiments(data_module: AbstractDataModule, partial_classifier: partial, name: str, max_epochs: int,
                              project: str = "EMG Armband", save_dir: str = 'wandb_logs', k: int = 0):
    classifier = partial_classifier()
    logger = WandbLogger(project=project, name=name, save_dir=save_dir)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", logger=logger, num_sanity_val_steps=0)
    trainer.fit(model=classifier, datamodule=data_module)
    trainer.test(model=classifier, datamodule=data_module)
    wandb.finish()
    classifier.xgbmodel.save_model(os.path.join(MODELS_FOLDER, f'{name.replace(" ", "_").lower()}_{k}_fold.json'))
    data_module.next_fold()


def xgb_cross_val_experiments_file(data_module: AbstractDataModule, model_files: List[str], name: str,
                                   max_epochs: int, project: str = "EMG Armband", save_dir: str = 'wandb_logs',
                                   k_folds: int = 10, seed: int = 0, time_window: int = 150, time_step: int = 10):
    pl.seed_everything(seed, workers=True)
    parameters = data_module.get_data_parameters()
    model = OriginalModel(**parameters)
    for k in tqdm(range(k_folds)):
        model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER, model_files[k])))
        model.model = model.model[0:26]
        partial_classifier = partial(LightningXGBClassifier, model, data_module.num_classes, monitor='val_accuracy',
                                     tree_method='gpu_hist', time_window=time_window, time_step=time_step)
        xgb_cross_val_experiments(data_module=data_module, partial_classifier=partial_classifier, name=name,
                                  max_epochs=max_epochs, project=project, save_dir=save_dir, k=k)
