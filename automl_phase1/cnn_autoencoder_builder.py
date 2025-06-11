import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape

def cnn_autoencoder_builder(params):
    input_layer = Input(shape=(28,28,1))
    x = input_layer
    encoder_layers = []

    for i in range(params['num_conv_layers']):
        conv = Conv2D(params['filters'], kernel_size=3, activation=params['activation'], padding='same', strides=2, name=f"encoder_conv_{i+1}")
        x = conv(x)
        encoder_layers.append(conv)

    shape_before_flatten = x.shape[1:]
    x = Flatten()(x)
    latent = Dense(params['latent_dim'], activation=params['activation'])(x)

    x = Dense(np.prod(shape_before_flatten), activation=params['activation'])(latent)
    x = Reshape(shape_before_flatten)(x)

    for i in range(params['num_conv_layers']):
        x = Conv2DTranspose(params['filters'], kernel_size=3, activation=params['activation'], padding='same', strides=2, name=f"decoder_deconv_{i+1}")(x)

    output_layer = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder_layers
