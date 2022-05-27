from datasets.data_module import AbstractDataModule
from models.Classifier import Classifier
import pytorch_lightning as pl
import os


def main():
    model = Classifier(8)
    data_module = AbstractDataModule(
        os.path.join('..', '..', 'Data', 'CapgMyo', 'CapgMyo.csv')
    )
    trainer = pl.Trainer(gpus=-1, max_epochs=4)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
