import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from automl_search import AutoMLSearch
from cnn_autoencoder_builder import cnn_autoencoder_builder
from get_plot_filters import plot_layer_filters, list_conv_layers

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


(x_train, _), (x_test, _) = mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)
x = x.astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))
X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)

param_grid = {
    'num_conv_layers': [2],
    'conv_filters_list': [[16, 32]],
    'kernel_sizes_list': [[(3,3), (3,3)]],
    'pooling_type': ['max'],
    'dropout_rate': [0.2],
    'dense_units': [32],
    'activation': ['relu'],
    'latent_dim': [8]
}

search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder,
    param_grid=param_grid,
    input_shape=(28,28,1),
    epochs=1
)

search.fit(X_train, X_val)

assert search.get_best_model() is not None
assert search.get_best_params() is not None

filters = search.extract_filters()
assert len(filters) > 0

print("✅ All tests passed.")
