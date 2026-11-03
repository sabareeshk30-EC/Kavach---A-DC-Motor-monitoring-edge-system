import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MODEL_PATH,
    QUANTIZED_MODEL_PATH,
    N_MELS,
    SAMPLE_RATE,
    CLIP_DURATION,
    HOP_LENGTH,
)


def run_benchmark(model_path_to_use, iterations=50, warmup=10):
    print(f"\n{'='*60}")
    print(f"BENCHMARKING MODEL: {model_path_to_use.name}")
    print(f"{'='*60}")

    available_providers = ort.get_available_providers()
    print(f"Detected available providers: {available_providers}")

    session = None
    active_provider = "CPU fallback"

    if "QNNExecutionProvider" in available_providers:
        try:
            session = ort.InferenceSession(
                str(model_path_to_use),
                providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"})],
            )
            active_provider = "QNN HTP"
            print("Successfully initialized session with QNN HTP")
        except Exception as e:
            print(f"QNN HTP initialization failed ({e}), falling back to CPU.")

    if session is None:
        session = ort.InferenceSession(
            str(model_path_to_use),
            providers=["CPUExecutionProvider"],
        )
        active_provider = "CPU fallback"
        print("Initialized session with CPU fallback")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model Input: '{input_name}', Expected Shape: {input_shape}")

    time_frames = int((SAMPLE_RATE * CLIP_DURATION) / HOP_LENGTH) + 1
    dummy_input = np.random.randn(1, 1, N_MELS, time_frames).astype(np.float32)

    # Warm-up
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark timed loop
    print(f"Running {iterations} benchmark iterations...")
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    print(f"\n--- Benchmark Results [{active_provider}] ---")
    print(f"Iterations:        {iterations}")
    print(f"Mean Latency:      {latencies.mean():.2f} ms")
    print(f"Median Latency:    {np.median(latencies):.2f} ms")
    print(f"P95 Latency:       {np.percentile(latencies, 95):.2f} ms")
    print(f"Min Latency:       {latencies.min():.2f} ms")
    print(f"Max Latency:       {latencies.max():.2f} ms")
    print(f"Throughput:        {1000.0 / latencies.mean():.1f} inferences/sec")
    print(f"{'='*60}\n")


def main():
    target = None
    if QUANTIZED_MODEL_PATH.exists():
        target = QUANTIZED_MODEL_PATH
    elif MODEL_PATH.exists():
        target = MODEL_PATH
    else:
        print("[ERROR] Neither base nor quantized ONNX model found.")
        print("Please run: python scripts/export_onnx.py first.")
        sys.exit(1)

    run_benchmark(target)


if __name__ == "__main__":
    main()
