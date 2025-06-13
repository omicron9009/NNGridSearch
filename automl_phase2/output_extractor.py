# phase2_output_extractor.py

import numpy as np

def extract_and_print_filters(model):
    """
    Extracts and prints filter shapes and weights layer-wise for all Conv2D layers.

    Args:
        model: Trained Keras model.
    """
    print("\n Extracting Filters from Model...\n")
    
    for idx, layer in enumerate(model.layers):
        if 'conv2d' in layer.name.lower():
            try:
                filters, biases = layer.get_weights()
                shape = filters.shape  # (3, 3, in_channels, out_channels)
                
                print(f"Layer {idx} - {layer.name}")
                print(f"  Filter Shape: {shape}")
                
                # Print first filter as example
                print(f"\n  Filter 1 weights (kernel values):\n")
                print(np.round(filters[:, :, :, 0], 4))  # only print 1st filter

                print("-" * 50)
            except:
                print(f" No filters found in layer: {layer.name}")
