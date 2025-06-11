# get_plot_filters.py

import matplotlib.pyplot as plt
import numpy as np
import math

def plot_layer_filters(model, layer_name=None, layer_index=None):
    """
    Plots all filters of a convolutional layer in a grid with simple inferences.

    Args:
        model: Trained Keras model.
        layer_name: Name of the layer (optional).
        layer_index: Index of the layer (optional).
    """
    if layer_name:
        layer = model.get_layer(name=layer_name)
    elif layer_index is not None:
        layer = model.layers[layer_index]
    else:
        raise ValueError("Provide either layer_name or layer_index")

    filters, biases = layer.get_weights()
    num_filters = filters.shape[-1]

    cols = 8
    rows = math.ceil(num_filters / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()

    for i in range(num_filters):
        f = filters[:, :, :, i]

        if f.shape[-1] == 1:
            f = f[:, :, 0]

        axs[i].imshow(f, cmap='viridis')
        axs[i].set_title(f'Filter {i+1}', fontsize=8)
        axs[i].axis('off')
    # Hide remaining axes if any
    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(f"Layer: {layer.name} | Total Filters: {num_filters}", fontsize=14)
    plt.tight_layout()
    plt.show()


def list_conv_layers(model):
    """
    Utility function to list all Conv2D layers in the model.
    """
    print("Available Conv2D layers:")
    for idx, layer in enumerate(model.layers):
        if 'conv2d' in layer.name.lower():
            print(f"Index: {idx}, Name: {layer.name}")
