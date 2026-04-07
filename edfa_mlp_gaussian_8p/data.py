import numpy as np
import os


def load_data(data_dir):
    """Load train/val arrays and optional labels + scalers from data_dir."""
    def _load(fname):
        return np.load(os.path.join(data_dir, fname))

    train_X = _load("train_X.npy")
    train_y = _load("train_y.npy")
    val_X   = _load("val_X.npy")
    val_y   = _load("val_y.npy")

    labels_path  = os.path.join(data_dir, "train_labels.npy")
    scalers_path = os.path.join(data_dir, "scalers_val.npy")

    labels  = np.load(labels_path,  allow_pickle=True) if os.path.exists(labels_path)  else None
    scalers = np.load(scalers_path, allow_pickle=True) if os.path.exists(scalers_path) else None

    return train_X, train_y, val_X, val_y, labels, scalers
