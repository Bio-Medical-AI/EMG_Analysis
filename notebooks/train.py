from datasets.abstract_data_module import AbstractDataModule
from models.ResNet import ResNet
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
import pickle


class Foo:
    @staticmethod
    def to_3_channels(self, data):
        return data.expand(-1, 3, -1, -1)


def main():
    pl.seed_everything(42, workers=True)

    model = ResNet(8, lr=0.1)
    logger = WandbLogger(project="EMG Armband", name="FirstTest")
    data_module = AbstractDataModule(
        os.path.join('..', '..', 'Data', 'CapgMyo', 'CapgMyo.csv'),
        batch_size=100
    )
    trainer = pl.Trainer(gpus=-1, max_epochs=28, logger=logger, accelerator="gpu")
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
