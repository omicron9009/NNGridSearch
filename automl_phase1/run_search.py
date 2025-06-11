from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np

from automl_search import AutoMLSearch
from cnn_autoencoder_builder import cnn_autoencoder_builder
from visualize_filters import print_filters
from get_plot_filters import plot_layer_filters, list_conv_layers

# Load dataset
(x_train, _), (x_test, _) = mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)
x = x.astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))
X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)
X_train, X_val = X_train[:10000], X_val[:2000]

# Param Grid
param_grid = {
    'num_conv_layers': [1, 2],     
    'filters': [8, 16],            
    'activation': ['relu'],        
    'latent_dim': [8, 16]          
}

# Build search
search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder,
    param_grid=param_grid,
    mode='random',
    scoring='val_loss',
    n_jobs=1,
    epochs=5
)

search.fit(X_train, X_val)

# Best Params
print("\nBest Hyperparameters Found:")
print(search.get_best_params())

# Extract filters & print numerically
filters = search.extract_filters()
print_filters(filters)

# Visualize filters
print("\nVisualizing filters of the best model...")
best_model = search.get_best_model()

# List available convolutional layers
list_conv_layers(best_model)

# Plot filters from first convolutional layer (index 1 usually Conv2D)
plot_layer_filters(best_model, layer_index=1)
