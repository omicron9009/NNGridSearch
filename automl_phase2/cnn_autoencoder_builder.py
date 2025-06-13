import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape

from tensorflow.keras import layers, models

def cnn_autoencoder_builder(input_shape, params):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Encoder
    for i in range(params['num_conv_layers']):
        filters = params['conv_filters_list'][i]
        kernel_size = params['kernel_sizes_list'][i]
        x = layers.Conv2D(filters, kernel_size, activation=params['activation'], padding='same')(x)

        if params['pooling_type'] == 'max':
            x = layers.MaxPooling2D(pool_size=(2,2))(x)
        elif params['pooling_type'] == 'avg':
            x = layers.AveragePooling2D(pool_size=(2,2))(x)

        if params['dropout_rate'] > 0:
            x = layers.Dropout(params['dropout_rate'])(x)

    x = layers.Flatten()(x)
    x = layers.Dense(params['dense_units'], activation=params['activation'])(x)
    encoded = layers.Dense(params['latent_dim'], activation=params['activation'])(x)

    # Decoder
    x = layers.Dense(params['dense_units'], activation=params['activation'])(encoded)
    x = layers.Dense(input_shape[0]*input_shape[1]*input_shape[2], activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(x)

    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

