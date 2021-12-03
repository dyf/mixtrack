import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def downconv_block(x, nf, ks):
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same')(x)
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, strides=2, padding='same')(x)    
    return x

def upconv_block(x, skipx, nf, ks):
    x = tfkl.Conv1DTranspose(filters=nf, kernel_size=ks, strides=2, padding='same')(x)                
    x = tfkl.Concatenate()([x, skipx])
    x = tfkl.Conv1D(filters=nf, kernel_size=ks, padding='same')(x)        
    return x

def unet(input_shape, num_layers, minf=20, ks=5):
    in_x = tfk.Input(shape=input_shape)
    
    x = in_x
    layer_outputs = [x]
    for li in range(num_layers):
        xout = downconv_block(x, minf*2**li, ks)
        layer_outputs.append(xout)
        x = xout
    
    y1 = x
    y2 = x
    for li in range(num_layers-1,-1,-1):
        y1 = upconv_block(y1, layer_outputs[li], minf*2**li, ks)

    for li in range(num_layers-1,-1,-1):
        y2 = upconv_block(y2, layer_outputs[li], minf*2**li, ks)

    y1 = tfkl.Conv1D(filters=2, kernel_size=ks, padding='same', name='vocals')(y1)
    y2 = tfkl.Conv1D(filters=2, kernel_size=ks, padding='same', name='accompaniment')(y2)

    return tfk.Model(inputs=in_x, outputs=[y1,y2])

if __name__ == "__main__":
    model = unet((1000000,2), 4)
    print(model.summary())