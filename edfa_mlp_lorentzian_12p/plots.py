import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def generate_plots(history, val_y, y_pred, config):
    """Generate 4 diagnostic plots and save to config.PLOTS_DIR."""
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    base = os.path.join(config.PLOTS_DIR, config.DATASET_NAME)

    # ── 1. Training history (loss + MAE) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{config.DATASET_NAME} - Training History", fontsize=13)
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history["mae"],     label="Train")
    axes[1].plot(history.history["val_mae"], label="Val")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    _save(fig, f"{base}_history.png")

    # ── 2. Predictions vs actual (first 6 outputs, 100 samples) ─
    n_show = min(6, val_y.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{config.DATASET_NAME} - Predictions vs Actual", fontsize=13)
    for i, ax in enumerate(axes.flatten()[:n_show]):
        ax.plot(val_y[:100, i],  label="Actual",    alpha=0.8)
        ax.plot(y_pred[:100, i], label="Predicted", alpha=0.8)
        ax.set_title(f"Output {i+1}")
        ax.legend()
        ax.grid(True)
    _save(fig, f"{base}_predictions.png")

    # ── 3. Scatter: predicted vs actual ─────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{config.DATASET_NAME} - Scatter (Predicted vs Actual)", fontsize=13)
    for i, ax in enumerate(axes.flatten()[:n_show]):
        ax.scatter(val_y[:, i], y_pred[:, i], alpha=0.3, s=5)
        mn = min(val_y[:, i].min(), y_pred[:, i].min())
        mx = max(val_y[:, i].max(), y_pred[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Output {i+1}")
        ax.grid(True)
    _save(fig, f"{base}_scatter.png")

    # ── 4. Error distribution ────────────────────────────────────
    errors = val_y - y_pred
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{config.DATASET_NAME} - Error Distribution", fontsize=13)
    for i, ax in enumerate(axes.flatten()[:n_show]):
        ax.hist(errors[:, i], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.set_title(f"Output {i+1} Error")
        ax.grid(True)
    _save(fig, f"{base}_error.png")

    print(f"All plots saved to {config.PLOTS_DIR}/")
