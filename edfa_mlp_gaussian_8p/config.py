# ============================================================
#  Configuration for EDFA MLP - gaussian_8p
# ============================================================

DATASET_NAME = "gaussian_8p"
DATA_DIR = "../gaussian/dataset_output_labor(gaussian_8p)"

INPUT_DIM  = 401
OUTPUT_DIM = 24        # 8 peaks × 3 parameters
PEAK_COUNT = 8

EPOCHS        = 300
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
PATIENCE      = 30
MIN_DELTA     = 1e-5
DROPOUT_RATE  = 0.2

MODEL_DIR   = "models"
PLOTS_DIR   = "plots"
REPORTS_DIR = "reports"
