# ============================================================
#  Configuration for EDFA MLP - pseudovoigt_10p
# ============================================================

DATASET_NAME = "pseudovoigt_10p"
DATA_DIR = "../pseodo_voigt/dataset_output_pseudo_voigt_10p"

# Model dimensions
INPUT_DIM  = 401
OUTPUT_DIM = 40        # 10 peaks x 4 parameters (amplitude, center, width, eta)
PEAK_COUNT = 10

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
