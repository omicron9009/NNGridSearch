# get_plot_filters.py

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_layer_filters(model=None, filters=None, layer_name=None, layer_index=None):
    """
    Plots filters of a convolutional layer.

    Args:
        model: Trained Keras model (optional if filters provided).
        filters: Tuple of (filters, biases) directly from extract_filters().
        layer_name: Name of layer (if model provided).
        layer_index: Index of layer (if model provided).
    """
    # Determine how filters are provided
    if filters:
        filters, biases = filters
    elif model:
        if layer_name:
            layer = model.get_layer(name=layer_name)
        elif layer_index is not None:
            layer = model.layers[layer_index]
        else:
            raise ValueError("Provide either layer_name or layer_index")

        try:
            filters, biases = layer.get_weights()
        except:
            raise ValueError("Selected layer has no filters.")

    num_filters = filters.shape[-1]
    kernel_shape = filters.shape[:3]

    cols = 8
    rows = math.ceil(num_filters / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()

    for i in range(num_filters):
        f = filters[:, :, :, i]
        if f.shape[-1] == 1:
            f = f[:, :, 0]
        axs[i].imshow(f, cmap='viridis')
        axs[i].set_title(f'Filter {i+1}', fontsize=8)
        axs[i].axis('off')

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"Filters | Shape: {kernel_shape} | Total Filters: {num_filters}", fontsize=12)
    plt.tight_layout()
    plt.show()


def list_conv_layers(model):
    """
    Utility to list all Conv2D layers in model.
    """
    print("Available Conv2D layers:")
    for idx, layer in enumerate(model.layers):
        if 'conv2d' in layer.name.lower():
            print(f"Index: {idx}, Name: {layer.name}, Filters: {layer.filters if hasattr(layer, 'filters') else 'N/A'}")
