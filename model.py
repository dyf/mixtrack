import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def downconv_block(x, nf, ks):    
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    #x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    #x = tfkl.LeakyReLU()(x)
    #x = tfkl.MaxPooling1D()(x)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    #x = tfkl.Dropout(0.1)(x)
    return x

def upconv_block(x, skipx, nf, ks):    
    x = tfkl.Conv1DTranspose(filters=nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    #x = tfkl.UpSampling2D((2,1))(x)    
    x = tfkl.Concatenate()([x, skipx])
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    #x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    #x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(0.1)(x)
    return x

def unet(input_shape, nfs, ks=9):
    num_layers = len(nfs)

    in_x = tfk.Input(shape=input_shape)
    
    x = in_x
    layer_outputs = [x]
    for i in range(num_layers-1):
        xout = downconv_block(x, nfs[i], ks)
        layer_outputs.append(xout)
        x = xout

    x = tfkl.Conv1D(filters=nfs[-1], kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Dropout(0.1)(x)    
    
    for li in range(num_layers-2,-1,-1):    
        x = upconv_block(x, layer_outputs[li], nfs[li], ks)

    x = tfkl.Conv1D(filters=2, kernel_size=ks, padding='same', kernel_initializer='he_normal')(x)
    x = tfkl.LeakyReLU()(x)
    x = tfkl.Conv1D(filters=2, kernel_size=1, name='target')(x)    

    return tfk.Model(inputs=in_x, outputs=x)

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


def stereo_stft(x, frame_length, frame_step):
    x = [ 
        tf.signal.stft(x[:,:,i], frame_length=frame_length, frame_step=frame_step) 
        for i in range(x.shape[-1])
    ]

    x = [
        tf.stack([tf.math.real(s), tf.math.imag(s)], axis=3) for s in x
    ]

    x = tf.concat(x, axis=3)

    return x

def stereo_inverse_stft(x, frame_length, frame_step):
    nch = x.shape[-1] // 2

    x = [
        tf.complex(real=x[:,:,:,2*i],imag=x[:,:,:,2*i+1]) for i in range(nch)
    ]
    
    x = [
        tf.signal.inverse_stft(s, frame_length=frame_length, frame_step=frame_step) 
        for s in x
    ]
    
    x = tf.stack(x, axis=2)

    return x

def stft_unet(input_shape, nfs, ks, frame_length, frame_step):
    scale = 100.0

    num_layers = len(nfs)

    in_x = tfk.Input(shape=input_shape)

    x = stereo_stft(in_x, frame_length=frame_length, frame_step=frame_step) / scale

    layer_outputs = [x]

    for i in range(num_layers-1):
        xout = downconv2d_block(x, nfs[i], ks)
        layer_outputs.append(xout)
        x = xout

    x = tfkl.Conv2D(nfs[-1], kernel_size=ks, padding='same')(x)
    x = tfkl.LeakyReLU()(x)
        
    for li in range(num_layers-2,-1,-1):
        x = upconv2d_block(x, layer_outputs[li], nfs[li], ks)

    x = tfkl.Conv2D(4, kernel_size=(3,3), padding='same')(x)
    
    x = stereo_inverse_stft(x * scale, frame_length=frame_length, frame_step=frame_step)
    
    return tfk.Model(inputs=in_x, outputs=x)


if __name__ == "__main__":
    model = stft_unet((131072,2), nfs= [ 32, 64, 128, 256 ])
    #model = unet((1000000,2), nfs=[10,20,30,40,50])
    print(model.summary())