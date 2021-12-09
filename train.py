import os

import tensorflow as tf
import tensorflow.keras as tfk



import model as mixmodel
import dataset as mixdataset        

def train():
    N = 2048
    epochs = 100
    batch_size = 4
    seq_dur = 3.0
    target = 'drums'
    unet_filters = [ 30, 60, 90, 120, 150, 180, 210, 240 ]
    checkpoint_dir = "./training"        

    optimizer = tfk.optimizers.Adam(lr=0.0001)    

    model = mixmodel.unet((131072,2), unet_filters, ks=9)
    model.compile(loss='mse', optimizer=optimizer)
    print(model.summary())

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)    

    if manager.latest_checkpoint:
        print(f"restored from {manager.latest_checkpoint}")
    else:
        print("training from scratch")

    #ds = mixdataset.MusdbData('D:/MUSDB18/FullHQ', is_wav=True)
    ds = mixdataset.MusdbData('D:/MUSDB18/Full', is_wav=False, target=target)
    train_ds = ds.random_dataset(N, seq_dur, subset='train')
    val_ds = ds.random_dataset(batch_size*16, seq_dur, subset='val', augment=False)
        
    class CustomCallback(tfk.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            manager.save()

    model.fit(
        train_ds.batch(batch_size), 
        epochs=epochs, 
        validation_data=val_ds.batch(batch_size),
        callbacks=[CustomCallback()]
    )

def train_ft():
    N_tracks = 512
    samples_per_track = 8
    epochs = 100
    batch_size = 16
    seq_dur = 3.0
    target = 'drums'
    unet_filters =[ 32, 64, 128, 256, 512 ]
    checkpoint_dir = "./training_ft"        

    optimizer = tfk.optimizers.Adam(lr=0.0001)

    model = mixmodel.stft_unet((131072,2), unet_filters, ks=(3,3), frame_length=512, frame_step=512)
    model.compile(loss='mse', optimizer=optimizer)
    print(model.summary())

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)    

    if manager.latest_checkpoint:
        print(f"restored from {manager.latest_checkpoint}")
    else:
        print("training from scratch")

    #ds = mixdataset.MusdbData('D:/MUSDB18/FullHQ', is_wav=True)
    ds = mixdataset.MusdbData('D:/MUSDB18/Full', is_wav=False, target=target)
    train_ds = ds.random_dataset(N_tracks, samples_per_track, seq_dur, subset='train')
    val_ds = ds.random_dataset(batch_size*4, samples_per_track, seq_dur, subset='val', augment=False)
        
    class CustomCallback(tfk.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            manager.save()

    model.fit(
        train_ds.batch(batch_size), 
        epochs=epochs, 
        validation_data=val_ds.batch(batch_size),
        callbacks=[CustomCallback()]
    )

if __name__ == "__main__": train_ft()