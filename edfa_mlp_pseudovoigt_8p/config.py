# ============================================================
#  Configuration for EDFA MLP - pseudovoigt_8p
# ============================================================

DATASET_NAME = "pseudovoigt_8p"
DATA_DIR = "../pseodo_voigt/dataset_output_pseudo_voigt_8p"

# Model dimensions
INPUT_DIM  = 401
OUTPUT_DIM = 32        # 8 peaks x 4 parameters (amplitude, center, width, eta)
PEAK_COUNT = 8

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
