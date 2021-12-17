import os

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

import model as mixmodel
import dataset as mixdataset        

def train():
    N_tracks = 512
    samples_per_track = 8
    epochs = 100
    batch_size = 4
    seq_dur = 3.0
    target = 'drums'
    unet_filters = [ 32, 64, 96, 128, 160, 192 ]
    checkpoint_dir = "./training"        

    optimizer = tfk.optimizers.Adam(learning_rate=0.0001)    

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
    train_ds = ds.random_dataset(N_tracks, samples_per_track, seq_dur, subset='train', augment=True)
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

def train_ft():
    N_tracks = 512
    samples_per_track = 8
    epochs = 100
    batch_size = 16
    seq_dur = 3.0
    target = 'drums'
    unet_filters =[ 32, 64, 128, 256, 512 ]
    checkpoint_dir = "./training_ft"       
    frame_length=512
    frame_step=512 

    optimizer = tfk.optimizers.Adam(lr=0.0001)

    model = mixmodel.stft_unet((131072,2), unet_filters, ks=(3,3), frame_length=frame_length, frame_step=frame_step)
    model.compile(loss=[None,'mse'], optimizer=optimizer)
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
    train_ds = ds.random_ft_dataset(N_tracks, samples_per_track, seq_dur, frame_length, frame_step, subset='train', augment=True)
    val_ds = ds.random_ft_dataset(batch_size*4, samples_per_track, seq_dur, frame_length, frame_step, subset='val', augment=False)
        
    class CustomCallback(tfk.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            manager.save()

    model.fit(
        train_ds.batch(batch_size), 
        epochs=epochs, 
        validation_data=val_ds.batch(batch_size),
        callbacks=[CustomCallback()]
    )

def train_gan():
    N_tracks = 512
    samples_per_track = 8
    epochs = 100
    batch_size = 8
    seq_dur = 3.0
    target = 'drums'
    unet_filters = [ 30, 60, 90, 120, 150, 180, 210, 240 ]
    disc_filters = [ 30, 60, 90, 120, 150, 180, 210, 240 ]
    checkpoint_dir = "./training_gan"       
    shape = (131072,2)
    
    disc_optimizer = tfk.optimizers.Adam(learning_rate=0.001)#, clipnorm=1, beta_1=0.5)    
    gan_optimizer = tfk.optimizers.Adam(learning_rate=0.001)#, clipnorm=1, beta_1=0.5)    
    
    discriminator = mixmodel.discriminator(shape, disc_filters, ks=9)
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy')

    unmixer = mixmodel.unet(shape, unet_filters, ks=9)
    discriminator.trainable = False    
    gan = mixmodel.dcgan(shape, unmixer, discriminator)
    #gan.compile(optimizer=gan_optimizer, loss=['mse','binary_crossentropy'])
    gan.compile(optimizer=gan_optimizer, loss=['mse','binary_crossentropy'], loss_weights=[0.1, 0.9])

    print(gan.summary())
    print(discriminator.summary())
    
    checkpoint = tf.train.Checkpoint(
        unmixer=unmixer,
        discriminator=discriminator,
        gan=gan,
        disc_optimizer=disc_optimizer,
        gan_optimizer=gan_optimizer
    )
        
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)    

    if manager.latest_checkpoint:
        print(f"restored from {manager.latest_checkpoint}")
    else:
        print("training from scratch")

    #ds = mixdataset.MusdbData('D:/MUSDB18/FullHQ', is_wav=True)
    ds = mixdataset.MusdbData('D:/MUSDB18/Full', is_wav=False, target=target)
    train_ds = ds.random_dataset(N_tracks, samples_per_track, seq_dur, subset='train', augment=True)
    val_ds = ds.random_dataset(batch_size*4, samples_per_track, seq_dur, subset='val', augment=False)

    for epoch in range(epochs):
        for i, (mixed, unmixed) in enumerate(train_ds.batch(batch_size)):            
            if i % 2 == 0:
                disc_loss_r, disc_loss_f = train_discriminator_step(mixed, unmixed, discriminator, unmixer)
                print(f"e:{epoch} b:{i} dlr:{disc_loss_r} dlf:{disc_loss_f}")
            else:
                gan_loss = train_gan_step(mixed, unmixed, gan)
                print(f"e:{epoch} b:{i} gl:{gan_loss}")

        manager.save()


def train_discriminator_step(mixed, unmixed, discriminator, unmixer):    
    y_real = np.ones(mixed.shape[0]) 
    y_fake = np.zeros(mixed.shape[0])
    
    unmixed_pred = unmixer.predict(mixed)    
    
    loss_real = discriminator.train_on_batch(unmixed, y_real)
    loss_fake = discriminator.train_on_batch(unmixed_pred, y_fake)

    return loss_real, loss_fake

def train_gan_step(mixed, unmixed, gan):    
    y_gan = np.ones(mixed.shape[0])    
    return gan.train_on_batch(mixed, [unmixed, y_gan])    


if __name__ == "__main__": train()