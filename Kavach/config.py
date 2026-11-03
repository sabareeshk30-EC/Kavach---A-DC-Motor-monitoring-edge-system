from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODEL_DIR / "motor_condition.onnx"
QUANTIZED_MODEL_PATH = MODEL_DIR / "motor_condition_int8.onnx"
BASELINE_PATH = MODEL_DIR / "healthy_baseline.json"

SAMPLE_RATE = 16000
CLIP_DURATION = 6

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal",
    "Imbalance/Rubbing Pattern",
    "Maintenance Check Recommended",
]

UNCERTAINTY_THRESHOLD = 0.60

ABNORMAL_WINDOWS_REQUIRED = 3
