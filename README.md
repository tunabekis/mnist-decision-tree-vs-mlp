# MNIST Classification: Decision Tree vs. MLP

A comparison of two machine learning approaches — Decision Trees and a
Multilayer Perceptron (MLP) — for handwritten digit classification on the
MNIST-784 dataset.

## Overview

The script draws a stratified 10,000-sample subset of MNIST, splits it into
train/validation/test sets (70% / 15% / 15%), and trains:

- **Decision Trees** (scikit-learn): a pre-pruned tree (`max_depth=5`) and an
  unconstrained tree, to illustrate the bias-variance tradeoff and
  overfitting.
- **MLPs** (PyTorch): a single-hidden-layer network trained with two
  different activation functions, **ReLU** and **Tanh**, to compare their
  effect on convergence speed and final accuracy.

All four models (2 trees + 2 MLPs) are evaluated on a held-out test set
using Accuracy, Precision, Recall, and F1-score (macro-averaged). Training
and validation loss curves are plotted for both MLP variants to visualize
overfitting behavior.

## Tech Stack

- Python 3
- [PyTorch](https://pytorch.org/) — MLP implementation and training
- [scikit-learn](https://scikit-learn.org/) — Decision Trees, data splitting, and metrics
- [Matplotlib](https://matplotlib.org/) — loss curve plotting

## Project Structure

```
hw2/Code/
├── main.py            # End-to-end training and evaluation pipeline
└── requirements.txt    # Python dependencies
```

## How to Run

```bash
pip install -r hw2/Code/requirements.txt
python hw2/Code/main.py
```

Running the script will:

1. Download MNIST-784 (via `sklearn.datasets.fetch_openml`) — requires
   internet access on first run.
2. Train and evaluate the Decision Tree and MLP models.
3. Print train/validation accuracy for the trees and final test metrics for
   all four models to the console.
4. Save and display a loss-curve plot (`loss_curves.png`) comparing the
   ReLU and Tanh MLPs.

## Notes

- The random seed is derived deterministically from a fixed student number
  (`seed = student_number % 200`) to ensure reproducible data sampling and
  splits.
