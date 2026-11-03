import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_audio
from config import (
    MODEL_PATH,
    QUANTIZED_MODEL_PATH,
    CLASS_NAMES,
    UNCERTAINTY_THRESHOLD,
)


class EdgeInference:

    def __init__(self, model_path: Path | str | None = None):
        self.provider = "CPU fallback"
        self.session = None

        if model_path is not None:
            resolved_model_path = str(model_path)
        elif QUANTIZED_MODEL_PATH.exists():
            resolved_model_path = str(QUANTIZED_MODEL_PATH)
        elif MODEL_PATH.exists():
            resolved_model_path = str(MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"No ONNX model found at {MODEL_PATH} or {QUANTIZED_MODEL_PATH}. "
                "Please run `python scripts/export_onnx.py` first."
            )

        self.model_path = resolved_model_path
        available = ort.get_available_providers()
        qnn_available = "QNNExecutionProvider" in available

        # Attempt QNN HTP initialization on Snapdragon NPU
        if qnn_available:
            qnn_provider_options = [
                {"backend_path": "QnnHtp.dll"},
                {"backend_path": "libQnnHtp.so"},
                {}
            ]
            for opts in qnn_provider_options:
                try:
                    providers = [("QNNExecutionProvider", opts)] if opts else ["QNNExecutionProvider"]
                    self.session = ort.InferenceSession(
                        resolved_model_path,
                        providers=providers,
                    )
                    self.provider = "QNN HTP"
                    break
                except Exception as error:
                    # Continue trying next provider option or fall back
                    pass

        # CPU fallback if QNN unavailable or session creation failed
        if self.session is None:
            self.session = ort.InferenceSession(
                resolved_model_path,
                providers=["CPUExecutionProvider"],
            )
            self.provider = "CPU fallback"

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, audio_path):
        features = preprocess_audio(audio_path)

        # Warm-up / execution timing
        start = time.perf_counter()

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: features},
        )

        latency = (time.perf_counter() - start) * 1000.0

        logits = outputs[0][0]
        probabilities = self._softmax(logits)

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        # Apply triage uncertainty threshold rule
        if confidence < UNCERTAINTY_THRESHOLD:
            predicted_index = 2  # Maintenance Check Recommended

        return {
            "class_id": predicted_index,
            "class_name": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "latency_ms": latency,
            "provider": self.provider,
            "raw_logits": logits.tolist(),
        }

    @staticmethod
    def _softmax(x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)
