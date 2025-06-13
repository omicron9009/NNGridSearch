# AutoML_Demo_Template_V2.py

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from automl_search import AutoMLSearch
from cnn_autoencoder_builder_v2 import cnn_autoencoder_builder_v2
from get_plot_filters import plot_all_filters, list_conv_layers

# 1️⃣ Load and prepare dataset (replaceable with any dataset)
(x_train, _), (x_test, _) = mnist.load_data()
x = np.concatenate((x_train, x_test), axis=0)
x = x.astype('float32') / 255.0
x = np.reshape(x, (x.shape[0], 28, 28, 1))
X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)

# Optional: subset to speed up testing
X_train, X_val = X_train[:5000], X_val[:1000]

# 2️⃣ Define parameter grid (fully flexible for your search space)
param_grid = {
    'num_conv_layers': [1, 2],
    'conv_filters_list': [[16], [16, 32]],
    'kernel_sizes_list': [[(3, 3)], [(3, 3), (3, 3)]],
    'pooling_type': ['max'],
    'dropout_rate': [0.2],
    'dense_units': [32],
    'activation': ['relu'],
    'latent_dim': [8]
}

# 3️⃣ Build and initialize AutoML search
search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder_v2,
    param_grid=param_grid,
    mode='random',  # or 'grid'
    scoring='val_loss',
    n_jobs=1,
    epochs=5
)

# 4️⃣ Run the search
search.fit(X_train, X_val)

# 5️⃣ Show best parameters
print("\n✅ Best Hyperparameters Found:")
print(search.get_best_params())

# 6️⃣ Extract filters from best model
filters = search.extract_filters()

# 7️⃣ List available conv layers
list_conv_layers(search.get_best_model())

# 8️⃣ Plot filters for inspection
plot_all_filters(filters)

print("\n🎯 AutoML V2 PRO Demo Run Completed Successfully!")
