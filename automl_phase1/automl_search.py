import itertools
import multiprocessing
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
import os


class AutoMLSearch:
    def __init__(self, model_builder, param_grid, mode='grid', scoring='val_loss', n_jobs=1, epochs=5, batch_size=128, log_dir='./logs'):
        self.model_builder = model_builder
        self.param_grid = param_grid
        self.mode = mode
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_dir = log_dir

        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_encoder_layers = None

    def _param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        grid = list(itertools.product(*values))

        if self.mode == 'random':
            grid = random.sample(grid, min(len(grid), 20))

        for instance in grid:
            yield dict(zip(keys, instance))

    def _train_model(self, args):
        params, X_train, X_val = args
        try:
            model, encoder_layers = self.model_builder(params)

            run_logdir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks = [
                EarlyStopping(patience=3, restore_best_weights=True),
                TensorBoard(log_dir=run_logdir)
            ]

            history = model.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                callbacks=callbacks
            )

            final_score = history.history[self.scoring][-1]
            print(f"Params: {params} | Score: {final_score:.4f}")

            return {
                'params': params,
                'score': final_score,
                'model': model,
                'encoder_layers': encoder_layers
            }
        except Exception as e:
            print(f"Failed on params {params}: {e}")
            return None

    def fit(self, X_train, X_val):
        param_list = list(self._param_combinations())
        tasks = [(params, X_train, X_val) for params in param_list]

        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                self.results = pool.map(self._train_model, tasks)
        else:
            self.results = [self._train_model(task) for task in tasks]

        self.results = [r for r in self.results if r is not None]
        self.results = sorted(self.results, key=lambda x: x['score'])

        if not self.results:
            raise RuntimeError("No successful model was trained. Please check param_grid or data.")

        self.best_model = self.results[0]['model']
        self.best_params = self.results[0]['params']
        self.best_encoder_layers = self.results[0]['encoder_layers']

    def get_best_model(self):
        return self.best_model

    def get_best_params(self):
        return self.best_params

    def extract_filters(self):
        filter_dict = {}
        for idx, conv_layer in enumerate(self.best_encoder_layers):
            filters, biases = conv_layer.get_weights()
            layer_filters = []
            for i in range(filters.shape[-1]):
                kernel = filters[:, :, 0, i]
                layer_filters.append(kernel)
            filter_dict[f"Layer_{idx+1}"] = layer_filters
        return filter_dict
