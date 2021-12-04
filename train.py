import model as mixmodel
import dataset as mixdataset
import tensorflow.keras as tfk

def train():
    N = 2048
    epochs = 20
    batch_size = 4
    unet_filters = [ 30, 60, 90, 120, 150, 180, 210 ]

    ds = mixdataset.MusdbData('D:/MUSDB18/Full')
    train_ds = ds.random_dataset(N, 1.0, subset='train')
    test_ds = ds.random_dataset(batch_size*4, 1.0, subset='test')

    model = mixmodel.unet((32768,2), unet_filters)
    model.compile(loss=['mse','mse'], optimizer=tfk.optimizers.Adam(lr=0.001))
    print(model.summary())

    my_callbacks = [
        tfk.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}.h5'),        
    ]

    model.fit(train_ds.batch(batch_size), epochs=epochs, validation_data=test_ds.batch(batch_size),callbacks=my_callbacks)

if __name__ == "__main__": train()