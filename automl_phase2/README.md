AutoML CNN Architecture Search — V2 PRO PACKAGE
🔥 Project Overview
This package implements an AutoML-based CNN Architecture Search system that automatically searches for optimal convolutional neural network (CNN) architectures for unsupervised feature extraction using Autoencoders.

The search automatically tunes:

Number of convolutional layers

Number of filters per layer

Kernel sizes

Latent dimension size

Activation functions

Dense layer units

Dropout regularization

Built fully modular to work on any image dataset.

🚀 What is New in V2 PRO
Feature	Improvement
✅ Full hyperparameter search	Multi-layer, multi-filter, multi-kernel
✅ Random or Grid search modes	Fully controlled
✅ Clean modular architecture	Easy to extend
✅ Filter extraction	Layer-wise filters easily accessible
✅ Filter plotting	Fully automated with filter labels
✅ Logging utilities	For safe search runs
✅ Simplified param grids	Very easy to configure
✅ Fully production-ready	Structured & documented

🔧 Folder Structure
graphql
Copy
Edit
AutoML_CNN_V2/
│
├── automl_search.py            # Full search engine logic
├── cnn_autoencoder_builder_v2.py # CNN Autoencoder model builder
├── run_search_v2.py            # Main runnable file
├── unit_test_v2.py             # Lightweight test suite
├── get_plot_filters.py         # Plotting & visualizing filters
├── utils/
│   └── logger.py               # Custom logging (optional)
│
├── README.md
├── requirements.txt
└── .gitignore
📊 Core Workflow
1️⃣ Load Dataset
You can use any image dataset (MNIST example provided).

2️⃣ Preprocess Data
Rescale and reshape into (height, width, channels) format.

3️⃣ Define Param Grid
Specify:

num_conv_layers: [1, 2, 3]

conv_filters_list: [[16], [32, 64]]

kernel_sizes_list: [[(3,3)], [(3,3),(3,3)]]

dropout_rate: [0.2]

dense_units: [32]

latent_dim: [8, 16]

4️⃣ Run Search
Use:

python
Copy
Edit
search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder_v2,
    param_grid=param_grid,
    mode='random', 
    scoring='val_loss',
    n_jobs=1,
    epochs=5
)
search.fit(X_train, X_val)
5️⃣ Extract Best Model & Filters

python
Copy
Edit
best_model = search.get_best_model()
filters = search.extract_filters()
6️⃣ Visualize Filters

python
Copy
Edit
from get_plot_filters import list_conv_layers, plot_layer_filters

list_conv_layers(best_model)
plot_layer_filters(model=best_model, layer_index=1)
⚙️ Autoencoder Explanation
Encoder compresses input images into a lower-dimensional latent space.

Decoder reconstructs the original images from this compressed representation.

Model learns efficient feature extraction automatically.

Why Autoencoders?
Because we're using unsupervised data without labels — so autoencoders force the network to extract useful features via reconstruction.

🎯 Inference Explanation for Filters
The filters learned at each layer correspond to basic pattern extractors.

Early layers: detect edges, gradients, textures.

Deeper layers: learn object parts, shapes, and higher-order abstractions.

By printing/plotting filters you can inspect how the network extracts features layer-by-layer.

📊 Filter Interpretation (Phase 2)
In Phase 2, the AutoML engine optimizes not only hyperparameters but also the network depth and filter configurations. After the search, we extract and analyze filters layer-wise:

Layer 1 Filters: (3, 3, 1, N)

First layer filters operate directly on raw input data (e.g., grayscale image channels).

They typically learn simple patterns like vertical/horizontal edges, corners, or blobs.

The weights contain small positive and negative values representing feature activations or suppressions.

Layer 2 Filters: (3, 3, previous_filters, N)

These filters operate on the feature maps produced by Layer 1.

Each kernel now has weights for all previous filters (e.g., 16 channels).

They capture more complex patterns such as combinations of curves, digit strokes, and texture patterns.

The filters provide interpretable insights into how the neural network progressively abstracts features from low-level to mid-level information.

📦 Example Run Template
python
Copy
Edit
from automl_search import AutoMLSearch
from cnn_autoencoder_builder_v2 import cnn_autoencoder_builder_v2

# load dataset...

search = AutoMLSearch(
    model_builder=cnn_autoencoder_builder_v2,
    param_grid=param_grid,
    mode='random',
    scoring='val_loss',
    n_jobs=1,
    epochs=5
)

search.fit(X_train, X_val)
print(search.get_best_params())
filters = search.extract_filters()
plot_layer_filters(model=search.get_best_model(), layer_index=1)
🔬 Requirements
bash
Copy
Edit
tensorflow
scikit-learn
matplotlib
numpy
bash
Copy
Edit
pip install -r requirements.txt
✅ Testing
You can run:

bash
Copy
Edit
python unit_test_v2.py
to verify correctness after installation.

⚠️ Notes for Future Versions (V3+)
EarlyStopping support

Auto refinement

Parallelized search

Dataset loaders for any type

Search acceleration techniques

🚀 Designed to be:
✅ Simple

✅ Modular

✅ Fully Expandable

✅ Dataset Agnostic

This is V2 PRO PACKAGE — Stable Base.