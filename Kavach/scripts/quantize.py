import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PATH,
    QUANTIZED_MODEL_PATH,
    DATA_DIR,
)
from preprocess import preprocess_audio


def collect_calibration_files():
    wav_files = []
    if DATA_DIR.exists():
        for ext in ("*.wav", "*.WAV"):
            wav_files.extend(list(DATA_DIR.rglob(ext)))
    return [str(p) for p in wav_files]


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Base ONNX model not found at: {MODEL_PATH}")
        print("Please run: python scripts/export_onnx.py first.")
        sys.exit(1)

    calibration_files = collect_calibration_files()

    if not calibration_files:
        print("[WARNING] No representative calibration audio files found in data/.")
        print("INT8 static quantization for Snapdragon QNN HTP/NPU requires representative real motor recordings")
        print("to calculate accurate activation calibration histograms and prevent distortion.")
        print("Please place representative .wav audio samples in data/ (e.g. data/normal/) and rerun.")
        sys.exit(0)

    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat,
        )

        class MotorCalibrationDataReader(CalibrationDataReader):
            def __init__(self, file_list):
                super().__init__()
                self.file_list = file_list
                self.index = 0

            def get_next(self):
                if self.index >= len(self.file_list):
                    return None
                path = self.file_list[self.index]
                self.index += 1
                try:
                    features = preprocess_audio(path)
                    return {"audio_features": features}
                except Exception as err:
                    print(f"[WARN] Failed to preprocess calibration sample {path}: {err}")
                    return self.get_next()

            def rewind(self):
                self.index = 0

        print(f"Found {len(calibration_files)} representative calibration samples. Starting INT8 static quantization...")
        reader = MotorCalibrationDataReader(calibration_files)

        QUANTIZED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        quantize_static(
            model_input=str(MODEL_PATH),
            model_output=str(QUANTIZED_MODEL_PATH),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        print(f"[SUCCESS] Quantized INT8 model exported to: {QUANTIZED_MODEL_PATH}")

    except ImportError as e:
        print(f"[ERROR] onnxruntime.quantization not available: {e}")
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}")


if __name__ == "__main__":
    main()
