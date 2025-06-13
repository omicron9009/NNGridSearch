# 🧠 AutoML Neural Network Search (Phase 2)

Welcome to the **AutoML Neural Network Architecture Search — Phase 2**.

This project builds a fully automated deep learning architecture search system designed to:

- 🔍 Automatically explore CNN + AutoEncoder structures.
- ⚙️ Handle flexible hyperparameter combinations.
- 🔢 Search across depth (layers) and width (filters, kernels, dense units, latent space).
- 🧮 Extract and print filters from trained convolutional layers.
- 🖼️ Visualize filters layer-wise (optional).
- 📊 Modular design: easy to extend for any image dataset.

---

## 🚀 Features

- **AutoML Search Engine**
  - Supports both `grid` and `random` search.
  - Random search uses internal clipping for faster execution.
  
- **Dynamic Model Builder**
  - Builds CNN AutoEncoders dynamically based on search parameters.

- **Filter Extraction**
  - Extracts weights (filters) from convolutional layers after training.

- **Filter Visualization**
  - Prints or plots filters to understand layer-wise feature extraction.

- **Completely Modular**
  - Simple file structure for easy extension and testing.

---

## ✅ Current Phase: Phase 2 PRO Version

- Multi-layer CNN AutoEncoders
- Fully parameterized search space
- Multiple convolutional layers handled automatically
- Clean filter extraction per layer
- Streamlined random search with sample clipping

---

## ⚠️ Note

> This project is intentionally simple, interpretable and fully transparent AutoML framework — perfect for:
> 
> - Learning
> - Experimentation
> - Extension to real-world datasets

---

## 🔧 Next Steps

- Build more advanced optimization (Phase 3)
- Add dataset auto-adaptation
- Implement scoring customization
- More efficient search strategies

---

