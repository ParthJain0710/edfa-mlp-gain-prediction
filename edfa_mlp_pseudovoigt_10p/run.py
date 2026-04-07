"""
Master script - EDFA MLP pipeline for pseudovoigt_10p
Run with:  python run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data   import load_data
from models import build_mlp
from train  import train_model
from eval   import evaluate_model
from plots  import generate_plots


def main():
    print(f"\n{'='*60}")
    print(f"  EDFA MLP Pipeline - {config.DATASET_NAME}")
    print(f"{'='*60}\n")

    here     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, config.DATA_DIR))

    print("Loading data ...")
    train_X, train_y, val_X, val_y, labels, scalers = load_data(data_dir)
    print(f"  Train : {train_X.shape}  ->  {train_y.shape}")
    print(f"  Val   : {val_X.shape}  ->  {val_y.shape}")

    print("\nBuilding model ...")
    model = build_mlp(config.INPUT_DIM, config.OUTPUT_DIM, config.DROPOUT_RATE)
    model.summary()

    print("\nTraining ...")
    history = train_model(model, train_X, train_y, val_X, val_y, config)

    print("\nEvaluating ...")
    report, y_pred = evaluate_model(model, val_X, val_y, config)
    print(f"  MSE : {report['overall_mse']:.6f}")
    print(f"  MAE : {report['overall_mae']:.6f}")
    print(f"  R²  : {report['overall_r2']:.6f}")

    print("\nGenerating plots ...")
    generate_plots(history, val_y, y_pred, config)

    print(f"\n{'='*60}")
    print(f"  Done!  Results saved to:")
    print(f"    Model   : {config.MODEL_DIR}/")
    print(f"    Plots   : {config.PLOTS_DIR}/")
    print(f"    Reports : {config.REPORTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
