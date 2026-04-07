# ============================================================
#  Configuration for EDFA MLP - lorentzian_12p
# ============================================================

DATASET_NAME = "lorentzian_12p"
DATA_DIR = "../lorentzian/dataset_output_labor(lorentzian_12p)"

# Model dimensions
INPUT_DIM  = 401
OUTPUT_DIM = 36        # 12 peaks × 3 parameters
PEAK_COUNT = 12

# Training hyper-parameters
EPOCHS        = 300
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
PATIENCE      = 30
MIN_DELTA     = 1e-5
DROPOUT_RATE  = 0.2

# Output directories (relative to this folder)
MODEL_DIR   = "models"
PLOTS_DIR   = "plots"
REPORTS_DIR = "reports"
