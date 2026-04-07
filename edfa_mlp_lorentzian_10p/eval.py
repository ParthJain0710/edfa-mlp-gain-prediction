import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, val_X, val_y, config):
    """Compute overall + per-output metrics; write a text report."""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    y_pred = model.predict(val_X, verbose=0)

    mse = mean_squared_error(val_y, y_pred)
    mae = mean_absolute_error(val_y, y_pred)
    r2  = r2_score(val_y, y_pred)

    per_mse = np.mean((val_y - y_pred) ** 2, axis=0)
    per_mae = np.mean(np.abs(val_y - y_pred), axis=0)
    per_r2  = [r2_score(val_y[:, i], y_pred[:, i]) for i in range(val_y.shape[1])]

    report_path = os.path.join(config.REPORTS_DIR, f"{config.DATASET_NAME}_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 55 + "\n")
        f.write(f" EDFA MLP Evaluation - {config.DATASET_NAME}\n")
        f.write("=" * 55 + "\n\n")
        f.write("Overall Metrics\n")
        f.write(f"  MSE : {mse:.8f}\n")
        f.write(f"  MAE : {mae:.8f}\n")
        f.write(f"  R²  : {r2:.8f}\n\n")
        f.write("Per-Output Metrics\n")
        for i in range(len(per_mse)):
            peak  = i // 3 + 1
            param = ["amplitude", "center", "width"][i % 3]
            f.write(
                f"  [{i+1:3d}] peak {peak:2d} {param:<10s} | "
                f"MSE={per_mse[i]:.6f}  MAE={per_mae[i]:.6f}  R²={per_r2[i]:.6f}\n"
            )

    print(f"Report saved -> {report_path}")

    return {
        "dataset":      config.DATASET_NAME,
        "overall_mse":  float(mse),
        "overall_mae":  float(mae),
        "overall_r2":   float(r2),
        "per_class_mse": per_mse.tolist(),
        "per_class_mae": per_mae.tolist(),
        "per_class_r2":  per_r2,
    }, y_pred
