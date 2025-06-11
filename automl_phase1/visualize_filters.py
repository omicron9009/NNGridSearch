import numpy as np

def print_filters(filters):
    for layer, kernels in filters.items():
        print(f"\n{layer}:")
        for idx, kernel in enumerate(kernels):
            print(f"Filter {idx+1}:")
            print(np.round(kernel, 4))  # Rounded for cleaner output
            print('-' * 30)
