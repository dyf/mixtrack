import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_addons as tfa
import tensorflow_addons.layers as tfal
import numpy as np

import dataset as mixdata

def downconv_block(x, nf, ks):    
    xres = tfkl.Conv1D(filters=nf, kernel_size=ks, strides=4, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.ReLU()(xres)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)

    #x = tfkl.Concatenate()([x, xres])
    #x = tfal.GELU()(x)

    return x

def upconv_block(x, nf, ks):    
    xres = tfkl.Conv1DTranspose(filters=nf, kernel_size=ks, strides=4, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.ReLU()(xres)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    
    #x = tfkl.Concatenate()([x, xres])
    #x = tfal.GELU()(x)
    x = tfkl.Dropout(0.1)(x)
    return x

def unet(input_shape, nfs, ks=9):
    num_layers = len(nfs)

    in_x = tfk.Input(shape=input_shape)
    
    x = in_x
    layer_outputs = [x]
    for i in range(num_layers):
        xout = downconv_block(x, nfs[i], ks)
        layer_outputs.append(xout)
        x = xout

    x = upconv_block(x, nfs[-2], ks)
    
    print(layer_outputs)
    for li in range(num_layers-1,0,-1):
        
        x = tfkl.Concatenate()([x, layer_outputs[li]])
        x = upconv_block(x, nfs[li], ks)

    x = tfkl.Conv1D(filters=2, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfal.GELU()(x)
    x = tfkl.Conv1D(filters=2, kernel_size=1, name='target')(x)    

    return tfk.Model(inputs=in_x, outputs=x, name="unet_unmixer")

def downconv2d_block(x, nf, ks):    
    x = tfkl.Conv2D(nf, kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Conv2D(nf, strides=(2,2), kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
    return x

def upconv2d_block(x, skipx, nf, ks):
    x = tfkl.Conv2DTranspose(nf, strides=(2,2), kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Cropping2D(cropping=((0,0),(0,1)))(x)
    x = tfkl.Concatenate()([x, skipx])
    x = tfkl.Conv2D(nf, kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(0.1)(x)
    return x

def discriminator(input_shape, nfs, ks):

    in_x = tfk.Input(shape=input_shape)
    
    x = in_x
    for i, nf in enumerate(nfs):
        x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', activation='relu')(x)
        x = tfkl.Conv1D(filters=nf, strides=2, kernel_size=ks, padding='same', activation='relu')(x)
        x = tfkl.Dropout(0.1)(x)

    x = tfkl.Flatten()(x)
    x = tfkl.Dense(1, activation='sigmoid')(x)

    return tfk.Model(inputs=in_x, outputs=x, name="discriminator")

def dcgan(input_shape, unmixer, discriminator):
    mixed_input = tfk.Input(shape=input_shape, name="mixed_input")    

    unmixed_output = unmixer(mixed_input)
        
    disc_output = discriminator(unmixed_output)

    return tfk.Model(inputs=mixed_input, outputs=[unmixed_output, disc_output], name="gan")

def stft_unet(input_shape, nfs, ks, frame_length, frame_step):
    scale = 100.0

    num_layers = len(nfs)

    in_x = tfk.Input(shape=input_shape)

    x = mixdata.stereo_stft(in_x, frame_length=frame_length, frame_step=frame_step) / scale

    layer_outputs = [x]

    for i in range(num_layers-1):
        xout = downconv2d_block(x, nfs[i], ks)
        layer_outputs.append(xout)
        x = xout

    x = tfkl.Conv2D(nfs[-1], kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
        
    for li in range(num_layers-2,-1,-1):
        x = upconv2d_block(x, layer_outputs[li], nfs[li], ks)

    stft_out = tfkl.Conv2D(4, kernel_size=(3,3), padding='same', name='stft')(x) * scale
    
    x = mixdata.stereo_inverse_stft(stft_out, frame_length=frame_length, frame_step=frame_step)
    
    return tfk.Model(inputs=in_x, outputs=[x, stft_out])

def predict(model, audio, output_file):            
    model_input_len = model.inputs[0].shape[1]
    track_len = audio.shape[0]
    
    output = np.zeros_like(audio)
    input_buf = np.zeros((1,model_input_len,2), dtype=output.dtype)

    for i in range(0, track_len, model_input_len):
        print(i)
        r0,r1 = i, min(i+model_input_len, track_len)        
        
        input_buf.fill(0)
        input_buf[0,:r1-r0,:] = audio[r0:r1,:]
        pred = model.predict(input_buf)
        output[r0:r1,:] = pred[0,:r1-r0,:]

    np.savez(output_file, audio=output)


if __name__ == "__main__":
    import musdb
    db = musdb.DB('D:/MUSDB18/Full', subsets=['test'])
    model = tfk.models.load_model('models/unet')
    drums = predict(model, db.tracks[4].audio, 'unmix_test.npz')