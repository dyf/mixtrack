import model as mixmodel
import dataset as mixdataset
def train():
    N = 32
    epochs = 10
    batch_size = 16

    ds = mixdataset.MusdbData('D:/MUSDB18/Full')
    train_ds = ds.random_dataset(N, 3.2, subset='train')
    test_ds = ds.random_dataset(N, 3.2, subset='test')

    model = mixmodel.unet((131072,2), 4)
    model.compile(loss=['mse','mse'], optimizer='adam')
    print(model.summary())
    model.fit(train_ds.batch(batch_size), epochs=epochs, validation_data=test_ds.batch(batch_size))
    model.save('model')

if __name__ == "__main__": train()