import pytorch_lightning as pl
import data as mixdata
import model as mixmodel

def train():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='../training_torch',
        monitor='train_loss',
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=True)
    dm =  mixdata.MusdbDataModule()
    
    model = mixmodel.UNetAudio()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__": train()