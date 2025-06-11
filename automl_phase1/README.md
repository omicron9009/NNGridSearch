AutoML Phase 1 Pro
Unsupervised CNN Autoencoder Search Engine with Filter Extraction

Overview:

This project implements an AutoML system that automatically tunes Convolutional Autoencoders using unsupervised learning. It performs hyperparameter search, trains multiple models, selects the best configuration based on validation loss, and extracts the learned 3x3 convolution filters from each encoder layer.

Key Features:

Automated hyperparameter search (number of layers, filters, activation functions, latent dimension)

Unsupervised learning using CNN autoencoders

Filter extraction and display per encoder layer

Unit testing to validate pipeline

Fully modular design

Project Structure:

automl_search.py — The main AutoML search engine that handles hyperparameter combinations, model training, evaluation, and best model selection.

cnn_autoencoder_builder.py — Builds CNN autoencoder models dynamically based on provided parameters.

run_search.py — Entry point script to run the full AutoML search pipeline.

visualize_filters.py — Prints layer-wise filters (numerical 3x3 weights) of the best model to the console.

unit_test.py — Lightweight unit test to validate the pipeline runs correctly on small sample data.

requirements.txt — Contains all the necessary Python packages for the project.

Installation:

Clone the repository:
git clone <your-repository-url>
cd automl_phase1_pro

(Optional) Create a virtual environment:
python -m venv venv
source venv/bin/activate (Linux/macOS)
venv\Scripts\activate (Windows)

Install dependencies:
pip install -r requirements.txt

Usage:

To run the full AutoML search:
python run_search.py

The script will:

Load MNIST dataset

Perform hyperparameter search

Display best hyperparameters

Extract and print 3x3 filters per layer

To run unit tests:
python unit_test.py

This performs a quick test of the full pipeline using a minimal parameter grid.

How It Works:

The system generates combinations of hyperparameters (number of convolution layers, filters, activation functions, latent space size).

For each combination, it builds a CNN autoencoder and trains it using unsupervised learning (reconstruction loss).

EarlyStopping prevents overfitting.

After all models are trained, the one with the lowest validation loss is selected as the best model.

The learned 3x3 filters of each encoder layer are extracted and printed.

Filter Interpretation:

After the AutoML search completes, the system extracts the trained 3x3 convolution filters from each encoder layer. These filters represent small learned patterns that the neural network uses to process the input images.

Each filter is essentially a 3x3 weight matrix.

These filters detect low-level features such as edges, corners, and simple textures.

For example, filters may activate when they see horizontal edges, vertical lines, or diagonal transitions in the image.

The encoder layers stack these filters hierarchically: earlier layers detect simpler features, while deeper layers learn more abstract patterns.

The printed filter weights help visualize what patterns the model focused on for reconstruction.

Because this system uses unsupervised learning (autoencoders), these filters emerge automatically without any labels or supervision.

Sample Output:

Layer_1:
Filter 1:
[[ 0.0142 0.0567 -0.0921]
[ 0.0735 -0.0183 0.0012]
[-0.0451 0.0054 0.0348]]
Filter 2:
[[ 0.0333 0.0425 -0.0102]
[-0.0563 0.0199 0.0123]
[ 0.0188 -0.0041 0.0097]]

Autoencoder Explanation:

An autoencoder is a type of neural network used for unsupervised learning. Its goal is to learn an efficient compressed representation of input data by attempting to reconstruct the input at its output.

An autoencoder consists of two main parts:

Encoder:

Compresses the input data into a smaller latent (compressed) representation.

Uses convolutional layers in this system to capture spatial patterns in images.

Learns important features like edges, shapes, or textures.

Decoder:

Attempts to reconstruct the original input data from the compressed representation.

Uses transposed convolutions or upsampling to bring the data back to original dimensions.

The closer the reconstruction is to the original input, the better the autoencoder has learned.

In this project, convolutional autoencoders are used, which are highly effective for image data like MNIST, because they preserve spatial relationships between pixels.

Since autoencoders don’t require any labels, the system learns directly from the raw input images, making this ideal for unsupervised AutoML optimization.

The AutoML search engine tunes hyperparameters of the autoencoder (number of convolution layers, filters, activation functions, and latent dimension size) to find the best model that reconstructs the input with minimum loss.


Why Use AutoML for Autoencoders?

Manually tuning autoencoder architectures is tedious and often suboptimal. This system uses AutoML to:

Automatically search for:

Optimal number of convolutional layers.

Number of filters per layer.

Activation functions.

Latent dimension size.

Evaluate reconstruction loss to select the best model.

Extract and display learned 3x3 convolution filters.

Eliminate manual trial-and-error for architecture design.

Unsupervised Learning

This system operates in an unsupervised learning setting because:

It does not require any labeled data.

The only objective is to reconstruct input images.

The network learns useful compressed features by itself.

Suitable for domains where labeled data is scarce.

Hyperparameter Search Space

The AutoML search currently explores the following hyperparameters:

num_conv_layers: Number of convolutional layers (1 or 2).
filters: Number of filters in each convolutional layer.
activation: Activation function (relu, sigmoid, etc).
latent_dim: Size of the latent compressed representation.

Example parameter grid:

param_grid = {
'num_conv_layers': [1, 2],
'filters': [16, 32],
'activation': ['relu', 'sigmoid'],
'latent_dim': [8, 16]
}

EarlyStopping


The training process uses EarlyStopping:

Monitors validation loss.

Stops training if no improvement is seen for 3 epochs.

Prevents overfitting and reduces unnecessary training time.

Limitations (Phase 1)

Search limited to simple convolutional autoencoders.

Kernel size fixed to 3x3.

No pooling layers or skip connections.

Uses only grid search and random search.

Training fully sequential (no parallelization yet).

Model evaluation purely based on reconstruction loss.



Next Phase (Planned Features)

Smarter search algorithms:

Bayesian Optimization

Hyperband

Successive Halving

Parallelized model evaluation.

Neural architecture search (NAS) for deeper networks.

Model visualization and scoring dashboards.

This completes Phase 1 Pro Version.




Roadmap:

Phase 1 (Completed): Core AutoML Search Engine with filter extraction.

Phase 2: Full Neural Architecture Search (NAS) including kernel size search, pooling, and deeper architectures.

Phase 3: Optimization algorithms (Bayesian optimization, reinforcement learning-based search).

Phase 4: Full production-grade multi-objective AutoML system.