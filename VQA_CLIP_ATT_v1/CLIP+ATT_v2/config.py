"""
Configuration cho VQA CLIP v2
"""

import os

# ===================
# PATHS
# ===================
DATA_DIR = "../data"
CHECKPOINT_DIR = "../checkpoints"  
RESULTS_DIR = "../results/clip_v2_experiments"

# ===================
# MODEL CONFIG
# ===================
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_ANSWERS = 1000
UNFREEZE_LAYERS = 2
DROPOUT_RATE = 0.3

# ===================
# TRAINING CONFIG
# ===================
BATCH_SIZE = 24
NUM_EPOCHS = 30
NUM_WORKERS = 4

# Learning rates
LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_CLIP = 2e-6
WEIGHT_DECAY = 1e-5

# Training techniques
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 0.001

# Scheduler
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'cosine_warm', 'reduce_plateau'
ETA_MIN = 1e-7

# Loss function
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0

# ===================
# AUGMENTATION CONFIG
# ===================
AUGMENT_CONFIG = {
    'random_resized_crop': {
        'size': 224,
        'scale': (0.8, 1.0),
        'ratio': (0.75, 1.333)
    },
    'random_horizontal_flip': {
        'p': 0.5
    },
    'color_jitter': {
        'brightness': 0.3,
        'contrast': 0.3,
        'saturation': 0.3,
        'hue': 0.1
    },
    'random_rotation': {
        'degrees': 10
    },
    'random_grayscale': {
        'p': 0.1
    }
}

# ===================
# EVALUATION CONFIG
# ===================
EVAL_BATCH_SIZE = 32
DEMO_SAMPLES = 5
ANALYSIS_SAMPLES = 1000

# ===================
# LOGGING CONFIG
# ===================
LOG_INTERVAL = 1
SAVE_CHECKPOINT_INTERVAL = 5
PLOT_INTERVAL = 5

def create_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [CHECKPOINT_DIR, RESULTS_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def print_config():
    """In configuration hiện tại"""
    print("⚙️ VQA CLIP v2 Configuration:")
    print("=" * 40)
    print(f"Model: {CLIP_MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Head LR: {LEARNING_RATE_HEAD}")
    print(f"CLIP LR: {LEARNING_RATE_CLIP}")
    print(f"Unfreeze layers: {UNFREEZE_LAYERS}")
    print(f"Scheduler: {SCHEDULER_TYPE}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Gradient clipping: {GRADIENT_CLIP_NORM}")
    print("=" * 40)
