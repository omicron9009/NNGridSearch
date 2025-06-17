from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

from automl_search import AutoMLSearch
from cnn_autoencoder_builder import cnn_autoencoder_builder
from visualize_filters import print_filters
from get_plot_filters import plot_layer_filters, list_conv_layers


# Load data
(x_train, _), (x_test, _) = mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)
x = x.astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))
X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)
X_train, X_val = X_train[:10000], X_val[:2000]

# Param grid
param_grid = {
    'num_conv_layers': [2, 3],
    'conv_filters_list': [
        [8],
        [8, 16],
        [8, 16, 32]
    ],
    'kernel_sizes_list': [
        [(3,3)],
        [(3,3), (3,3)],
        [(3,3), (3,3), (3,3)]
    ],
    'pooling_type': ['max', 'avg'],
    'dropout_rate': [0.1, 0.2, 0.3],
    'dense_units': [16, 32, 64],
    'activation': ['relu', 'tanh'],
    'latent_dim': [4, 8, 16]
}


# Search
search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder,
    param_grid=param_grid,
    input_shape=(28,28,1),
    mode='random',
    epochs=3
)

search.fit(X_train, X_val)

# Best params
print("Best Hyperparameters:")
print(search.get_best_params())

# Filters
filters = search.extract_filters()
print("Filters")
print("*"*50)
print_filters(filters)
print("*"*50)
best_model = search.get_best_model()
# list_conv_layers(best_model)
from output_extractor import extract_and_print_filters
extract_and_print_filters(best_model)
plot_layer_filters(model=best_model, layer_index=1)

# from output_extractor import extract_and_print_filters
# extract_and_print_filters(best_model)