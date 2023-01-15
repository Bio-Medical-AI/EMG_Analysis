from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, Specificity, Precision, F1Score
from datasets.frames import CapgMyoDataModule, MyoArmbandDataModule, NinaProDataModule
from definitions import MODELS_FOLDER
from models import SequenceClassifier, CRNN, Classifier, OriginalModel
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import Compose, ToTensor, Normalize
import warnings
from functools import partial
from utils import lstm_cross_val_experiment, xgb_cross_val_experiments_file, FIX_NDARRAY_TO_TENSOR_3D

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    max_epochs = 150
    max_seq_epochs = 150
    seed = 113
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])

    data_module_capgmyo = CapgMyoDataModule(
        batch_size=1000,
        k_folds=10,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
        seed=seed,
        shuffle_train=True
    )

    metrics = MetricCollection([Accuracy(average='micro', num_classes=data_module_capgmyo.num_classes),
                                Specificity(average='macro', num_classes=data_module_capgmyo.num_classes),
                                Precision(average='macro', num_classes=data_module_capgmyo.num_classes),
                                F1Score(average='macro', num_classes=data_module_capgmyo.num_classes)]).to(
        torch.device("cuda", 0))

    callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                 partial(EarlyStopping, monitor='val/Accuracy', patience=7, mode='max')]

    seq_callbacks = [partial(ModelCheckpoint, monitor='val/loss', dirpath=MODELS_FOLDER),
                     partial(EarlyStopping, monitor='val/Accuracy', patience=23, mode='max')]

    partial_classifier = partial(Classifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001},
                                 monitor='val/Accuracy', sched_kwargs={'patience': 4, 'mode': 'max'},
                                 time_window=[40, 150], time_step=[1, 1], metrics=metrics)
    partial_seq_classifier = partial(SequenceClassifier, optim_kwargs={'lr': 0.001, 'weight_decay': 0.001},
                                     monitor='val/Accuracy', sched_kwargs={'patience': 17, 'mode': 'max'})
    sequence_model = partial(CRNN, num_layers=1, dropout=0.25, hidden_size=128)
    lstm_cross_val_experiment(data_module=data_module_capgmyo, partial_classifier=partial_classifier,
                              partial_seq_classifier=partial_seq_classifier, name="CRNN CapgMyo",
                              max_epochs=max_epochs, max_seq_epochs=max_seq_epochs, sequence_model=sequence_model,
                              callbacks=callbacks, seq_callbacks=seq_callbacks, seed=seed, model_checkpoint_index=0)


if __name__ == '__main__':
    main()
