"""
MNIST digit classification: Decision Tree vs. Multilayer Perceptron (MLP).

Trains and compares two families of classifiers on a stratified 10k-sample
subset of MNIST-784:
  - Decision Trees (pre-pruned vs. unconstrained) via scikit-learn.
  - MLPs with different activation functions (ReLU vs. Tanh) via PyTorch.

Final performance is reported on a held-out test set using accuracy,
precision, recall, and F1-score.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

STUDENT_NUMBER = 11404148014
SUBSET_SIZE = 10000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PRUNED_TREE_DEPTH = 5
MLP_HIDDEN_UNITS = 128
MLP_EPOCHS = 20
MLP_BATCH_SIZE = 64
MLP_LEARNING_RATE = 0.001


def get_seed(student_number: int) -> int:
    """Derive the assignment-mandated random seed from the student number."""
    return student_number % 200


def set_global_seed(seed: int) -> None:
    """Seed PyTorch (CPU and CUDA) for reproducible model initialization/training."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_mnist_subset(seed: int, subset_size: int = SUBSET_SIZE):
    """Download MNIST-784 and draw a stratified, normalized subset of it."""
    print("Downloading MNIST-784...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x_all, y_all = mnist.data, mnist.target.astype(int)

    x_subset, _, y_subset, _ = train_test_split(
        x_all, y_all, train_size=subset_size, random_state=seed, stratify=y_all
    )

    # Flatten is already implicit in the 784-column layout; just normalize to [0, 1].
    x_subset = x_subset.astype("float32") / 255.0
    return x_subset, y_subset


def split_dataset(x, y, seed: int):
    """Stratified train/val/test split according to TRAIN/VAL/TEST_RATIO."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=seed, stratify=y
    )
    val_share_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(1 - val_share_of_temp), random_state=seed, stratify=y_temp
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def train_decision_trees(x_train, y_train, seed: int):
    """Train a pre-pruned and an unconstrained decision tree, for overfitting comparison."""
    pruned = DecisionTreeClassifier(max_depth=PRUNED_TREE_DEPTH, random_state=seed).fit(
        x_train, y_train
    )
    unconstrained = DecisionTreeClassifier(random_state=seed).fit(x_train, y_train)
    return pruned, unconstrained


def report_tree_fit(model, name: str, x_train, y_train, x_val, y_val) -> None:
    """Print train/validation accuracy for a fitted tree model."""
    print(
        f"{name} -> Train: {accuracy_score(y_train, model.predict(x_train)):.4f}, "
        f"Val: {accuracy_score(y_val, model.predict(x_val)):.4f}"
    )


def build_mlp(activation_fn: nn.Module, input_dim: int = 784, num_classes: int = 10) -> nn.Sequential:
    """Build a single-hidden-layer MLP with the given activation function."""
    return nn.Sequential(
        nn.Linear(input_dim, MLP_HIDDEN_UNITS),
        activation_fn,
        nn.Linear(MLP_HIDDEN_UNITS, num_classes),
    )


def train_mlp(model: nn.Sequential, train_loader: DataLoader, val_loader: DataLoader, name: str):
    """Train an MLP while tracking per-epoch train/validation loss (for overfitting analysis)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    train_losses, val_losses = [], []
    print(f"Training MLP ({name})...")

    for _ in range(MLP_EPOCHS):
        model.train()
        running_train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                running_val_loss += criterion(model(data), target).item()

        train_losses.append(running_train_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))

    return model, train_losses, val_losses


def plot_loss_curves(t_relu, v_relu, t_tanh, v_tanh, save_path: str = "loss_curves.png") -> None:
    """Plot train vs. validation loss for both MLP variants and save the figure."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t_relu, label="Train")
    plt.plot(v_relu, label="Val")
    plt.title("ReLU Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_tanh, label="Train")
    plt.plot(v_tanh, label="Val")
    plt.title("Tanh Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def evaluate_model(model, x, y, name: str, is_torch: bool = False) -> None:
    """Print Accuracy/Precision/Recall/F1 (macro-averaged) for a fitted model on (x, y)."""
    if is_torch:
        model.eval()
        with torch.no_grad():
            _, preds = torch.max(model(torch.Tensor(x)), 1)
            preds = preds.numpy()
    else:
        preds = model.predict(x)

    acc = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="macro")
    print(f"[{name}] Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")


def main() -> None:
    seed = get_seed(STUDENT_NUMBER)
    print(f"Random Seed: {seed}")
    set_global_seed(seed)

    x_subset, y_subset = load_mnist_subset(seed)
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_subset, y_subset, seed)

    # --- Decision Trees ---
    print("\n--- Decision Tree Results ---")
    dt_pruned, dt_unconstrained = train_decision_trees(x_train, y_train, seed)
    report_tree_fit(dt_pruned, f"Pre-pruned (depth={PRUNED_TREE_DEPTH})", x_train, y_train, x_val, y_val)
    report_tree_fit(dt_unconstrained, "Unconstrained", x_train, y_train, x_val, y_val)

    # --- MLPs (ReLU vs. Tanh) ---
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train)),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.Tensor(x_val), torch.LongTensor(y_val)), batch_size=MLP_BATCH_SIZE
    )

    mlp_relu, t_relu, v_relu = train_mlp(build_mlp(nn.ReLU()), train_loader, val_loader, "ReLU")
    mlp_tanh, t_tanh, v_tanh = train_mlp(build_mlp(nn.Tanh()), train_loader, val_loader, "Tanh")

    plot_loss_curves(t_relu, v_relu, t_tanh, v_tanh)

    # --- Final Evaluation on the Test Set ---
    print("\n--- Final Test Metrics ---")
    evaluate_model(dt_pruned, x_test, y_test, "Pruned DT")
    evaluate_model(dt_unconstrained, x_test, y_test, "Unconstrained DT")
    evaluate_model(mlp_relu, x_test, y_test, "MLP (ReLU)", is_torch=True)
    evaluate_model(mlp_tanh, x_test, y_test, "MLP (Tanh)", is_torch=True)


if __name__ == "__main__":
    main()
