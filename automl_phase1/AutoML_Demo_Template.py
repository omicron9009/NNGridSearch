# AutoML_Demo_Template.py

"""
AutoML Neural Network Hyperparameter Search - Robust Usage Template

This template demonstrates how to use your AutoMLSearch class end-to-end 
on any dataset with minimal setup.
"""

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from automl_search import AutoMLSearch
from cnn_autoencoder_builder import cnn_autoencoder_builder
from extract_filters import print_filters
from get_plot_filters import plot_layer_filters, list_conv_layers

# 1️ Load and preprocess dataset (MNIST used as example)
print("Loading dataset...")
(x_train, _), (x_test, _) = mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)
x = x.astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))

# Optional: Subsample for faster prototyping
X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)
X_train, X_val = X_train[:10000], X_val[:2000]

# 2️ Define hyperparameter search space
param_grid = {
    'num_conv_layers': [1, 2, 3],
    'filters': [8, 16, 32],
    'activation': ['relu', 'tanh'],
    'latent_dim': [8, 16, 32]
}

# 3️ Initialize and run search
search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder,
    param_grid=param_grid,
    mode='random',         # 'grid' or 'random'
    scoring='val_loss',
    n_jobs=1,
    epochs=5               # Change epochs for full training
)

print("\nRunning AutoML Search...")
search.fit(X_train, X_val)

# 4️ Retrieve best model and parameters
print("\n Best Hyperparameters Found:")
print(search.get_best_params())

# 5️ Extract numeric filters
filters = search.extract_filters()
print_filters(filters)

# 6️ Visualize filters from best model
best_model = search.get_best_model()
list_conv_layers(best_model)
plot_layer_filters(best_model, layer_index=1)

# 7️ (Optional) Save model
# best_model.save('best_autoencoder_model.h5')
