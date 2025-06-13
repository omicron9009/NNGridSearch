import itertools
import multiprocessing
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
import os

import itertools
import random
import numpy as np

class AutoMLSearch:

    def __init__(self, model_builder, param_grid, input_shape, mode='grid', n_jobs=1, epochs=10, scoring='val_loss'):
        self.model_builder = model_builder
        self.param_grid = param_grid
        self.input_shape = input_shape
        self.mode = mode
        self.n_jobs = n_jobs
        self.epochs = epochs
        self.scoring = scoring
        self.results = []

    def _param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        # Convert nested lists to tuples for cartesian product
        processed_values = []
        for v in values:
            if isinstance(v[0], list): 
                processed_values.append([tuple(i) for i in v])
            else:
                processed_values.append(v)

        combos = list(itertools.product(*processed_values))

        for combo in combos:
            params = dict(zip(keys, combo))

            # Convert back to list where required
            if 'conv_filters_list' in params:
                params['conv_filters_list'] = list(params['conv_filters_list'])
            if 'kernel_sizes_list' in params:
                params['kernel_sizes_list'] = list(params['kernel_sizes_list'])
            yield params

    def fit(self, X_train, X_val):
        combinations = list(self._param_combinations())

        if self.mode == 'random':
            sample_size = min(20, len(combinations))  # Clip to 20 samples
            combinations = random.sample(combinations, sample_size)

        for params in combinations:
            try:
                model = self.model_builder(self.input_shape, params)
                history = model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=self.epochs, verbose=0)
                score = history.history[self.scoring][-1]
                self.results.append({'params': params, 'score': score, 'model': model})
                print(f"Params: {params} | Score: {score:.4f}")
            except Exception as e:
                print(f"Failed on params {params}: {e}")

        self.results.sort(key=lambda x: x['score'])
        self.best_model = self.results[0]['model'] if self.results else None
        self.best_params = self.results[0]['params'] if self.results else None

    def get_best_model(self):
        return self.best_model

    def get_best_params(self):
        return self.best_params

    def extract_filters(self):
        filters = []
        for layer in self.best_model.layers:
            if 'conv2d' in layer.name.lower():
                filters.append(layer.get_weights()[0])
        return filters

