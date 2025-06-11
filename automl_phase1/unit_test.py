import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from automl_search import AutoMLSearch
from cnn_autoencoder_builder import cnn_autoencoder_builder
from get_plot_filters import plot_layer_filters, list_conv_layers

def test_automl_search():
    # Load and preprocess dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    x = x.astype('float32') / 255.0
    x = np.reshape(x, (x.shape[0], 28, 28, 1))
    
    # Use smaller data subset for faster test
    X_train, X_val = train_test_split(x, test_size=0.2, random_state=42)
    X_train, X_val = X_train[:10000], X_val[:2000]

    # Simple test param grid
    param_grid = {
        'num_conv_layers': [1],
        'filters': [16],
        'activation': ['relu'],
        'latent_dim': [8]
    }

    # Initialize AutoML search
    search = AutoMLSearch(
        model_builder=cnn_autoencoder_builder,
        param_grid=param_grid,
        n_jobs=1,
        epochs=3
    )

    search.fit(X_train, X_val)

    # Assertions
    assert search.get_best_model() is not None
    assert search.get_best_params() is not None

    # Extract and verify filters
    filters = search.extract_filters()
    assert len(filters) > 0

    # Print filters (optional)
    for layer_name, layer_filters in filters.items():
        print(f"Layer: {layer_name}")
        for idx, f in enumerate(layer_filters):
            print(f"Filter {idx+1}:\n{f}\n")

    # Visualize filters (optional during test)
    best_model = search.get_best_model()
    list_conv_layers(best_model)
    plot_layer_filters(best_model, layer_index=1)

    print(" All tests passed.")

if __name__ == '__main__':
    test_automl_search()
