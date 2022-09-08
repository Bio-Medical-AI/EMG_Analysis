import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datasets.stored_dataframe import MyoArmbandDataModule
from models import Classifier, OriginalModel, LightningXGBClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision.transforms import Compose, ToTensor, Normalize


def main():
    pl.seed_everything(42, workers=True)
    model_myoarmband = OriginalModel(7, 1, 8)
    classifier = Classifier(model_myoarmband, optim_kwargs={'lr': 0.001, 'weight_decay': 0.0001},
                            monitor='val_accuracy', sched_kwargs={'patience': 4, 'mode': 'max'})
    logger_myoarmband = WandbLogger(project="EMG Armband", name="MyoArmband")
    transform = Compose([
        ToTensor(),
        Normalize(0, 1)
    ])
    data_module_myoarmband = MyoArmbandDataModule(
        batch_size=10000,
        num_workers=8,
        train_transforms=transform,
        val_transforms=transform,
        test_transforms=transform,
    )
    early_stop_callback = EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
    trainer_myoarmband = pl.Trainer(gpus=-1, max_epochs=50, logger=logger_myoarmband, accelerator="gpu",
                                    callbacks=[early_stop_callback])

    trainer_myoarmband.fit(model=classifier, datamodule=data_module_myoarmband)
    trainer_myoarmband.test(model=classifier, datamodule=data_module_myoarmband)

    # logger_ninapro = WandbLogger(project="EMG Armband", name="NinaProXGB")
    # model_ninapro = OriginalModel(52, 1, 10)
    # classifier = Classifier.load_from_checkpoint('C:\\Science\\EMG_Analysis\\notebooks\\EMG '
    #                                              'Armband\\imieq3ak\\checkpoints\\epoch=99-step=125600.ckpt',
    #                                              model=model_ninapro)
    # adjusted_model = classifier.model
    # adjusted_model.model = adjusted_model.model[0:26]
    # xgb_classifier = LightningXGBClassifier(adjusted_model, 8)
    # trainer_xgb = pl.Trainer(max_epochs=2, accelerator="cpu", logger=logger_ninapro, num_sanity_val_steps=0)
    # data_module_xgb = NinaProDataModule(
    #     batch_size=8000,
    #     num_workers=3
    # )
    # trainer_xgb.fit(model=xgb_classifier, datamodule=data_module_xgb)
    # trainer_xgb.test(model=xgb_classifier, datamodule=data_module_xgb)


if __name__ == '__main__':
    main()
