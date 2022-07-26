from datasets.capgmyo_data_module import CapgMyoDataModule
from models.ResNet import ResNet
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
import pickle


def main():
    pl.seed_everything(42, workers=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    model = ResNet(8, lr=0.1)
    logger = WandbLogger(project="EMG Armband", name="FirstTest")
    data_module = CapgMyoDataModule(
        batch_size=70,
        train_transforms=transform,
        test_transforms=transform,
        val_transforms=transform
    )
    trainer = pl.Trainer(gpus=-1, max_epochs=28, logger=logger, accelerator="gpu")
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    main()
