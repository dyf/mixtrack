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
    unet_filters = [ 20, 40, 60, 80, 100, 120, 140, 160 ] #, 180, 200 ]

    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    #ds = mixdataset.MusdbData('D:/MUSDB18/FullHQ', is_wav=True)
    ds = mixdataset.MusdbData('D:/MUSDB18/Full', is_wav=False, target=target)
    train_ds = ds.random_dataset(N, seq_dur, subset='train')
    val_ds = ds.random_dataset(batch_size*16, seq_dur, subset='val')

    model = mixmodel.unet((131072,2), unet_filters)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        tfk.models.load_model(latest)
    else:
        model.compile(loss='mse', optimizer=tfk.optimizers.Adam(lr=0.001))    

    my_callbacks = [ 
        tfk.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
        tfk.callbacks.EarlyStopping(patience=5)
    ]

    model.fit(train_ds.batch(batch_size), epochs=epochs, validation_data=val_ds.batch(batch_size),callbacks=my_callbacks)

if __name__ == "__main__": train()