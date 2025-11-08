import os
import torch

# ---------- Paths & I/O ----------
CSV_PATH = os.getenv('CSV_PATH', '~/projects/novel_deeplearning_approach/mlp/data/combined5.outlier2.csv')
NOISE_JSON_PATH = os.getenv('NOISE_JSON_PATH', '~/projects/novel_deeplearning_approach/mlp/data/noise_standard_deviations.json')
SAVE_DIR = os.getenv('SAVE_DIR', 'saved_models_different')

# ---------- Training hyperparameters ----------
BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 5e-4
HIDDEN_SIZES = [256, 128, 64]
OUTPUT_SIZE = 1
DROPOUT = 0.2

NUDGE_FACTOR = 0.1
AUGMENT_PROB = 1.0
AUG_SEED = 42
CLIP_RANGE = None

USE_LAYERNORM = False
PRINT_EVERY = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
