import numpy as np

import numpy as np

def print_filters(filters):
    for idx, f in enumerate(filters):
        print(f"\nLayer {idx+1} filters shape: {f.shape}")
        for i in range(f.shape[-1]):
            kernel = f[..., i]
            print(f"\nFilter {i+1} weights:\n{kernel}")

