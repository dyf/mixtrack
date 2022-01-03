import pytorch_lightning as pl
import data as mixdata
import specunet 

def train():
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='../training_torch',
        monitor='loss',
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=True)
    
    dm =  mixdata.MusdbDataModule(batch_size=8, spectrogram=True)
        
    #model = mixmodel.MultiResUNet()
    model = specunet.SpectrogramUNet()

    trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__": train()