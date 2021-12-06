import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def downconv_block(x, nf, ks):    
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', activation='relu')(x)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, strides=2, padding='same', activation='relu')(x)
    x = tfkl.Dropout(0.1)(x)
    return x

def upconv_block(x, skipx, nf, ks):    
    x = tfkl.Conv1DTranspose(filters=nf, kernel_size=ks, strides=2, padding='same', activation='relu')(x)
    x = tfkl.Concatenate()([x, skipx])
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same', activation='relu')(x)
    #x = tfkl.Dropout(0.1)(x)
    return x

def unet(input_shape, nfs, minf=20, ks=7):
    num_layers = len(nfs)

    in_x = tfk.Input(shape=input_shape)
    
    x = in_x
    layer_outputs = [x]
    for i in range(num_layers-1):
        xout = downconv_block(x, nfs[i], ks)
        layer_outputs.append(xout)
        x = xout

    x = tfkl.Conv1D(filters=nfs[-1], kernel_size=ks, padding='same', activation='relu')(x)
    x = tfkl.Dropout(0.1)(x)    
    
    for li in range(num_layers-2,-1,-1):    
        x = upconv_block(x, layer_outputs[li], nfs[li], ks)

    x = tfkl.Conv1D(filters=2, kernel_size=ks, padding='same', name='target')(x)

    return tfk.Model(inputs=in_x, outputs=x)

if __name__ == "__main__":
    model = unet((1000000,2), nfs=[10,20,30,40,50])
    print(model.summary())