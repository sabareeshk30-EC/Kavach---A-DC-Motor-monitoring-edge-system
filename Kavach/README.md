# Kavach EdgeInspect

Offline edge-AI application for acoustic motor and fan condition triage (Normal, Imbalance/Rubbing Pattern, Maintenance Check Recommended). Engineered for offline execution targeting Qualcomm Snapdragon edge computing platforms with ONNX Runtime and automatic CPU fallback.

---

## Architecture

```text
Audio Input (.wav)
       │
       ▼
Preprocessing (16 kHz, mono, 6s fixed duration)
       │
       ▼
Log-Mel Spectrogram (64 mels, N_FFT=1024, Hop=256)
       │
       ▼
Lightweight CNN (Conv2D -> ReLU -> MaxPool -> AdaptiveAvgPool -> Linear)
       │
       ▼
ONNX Edge Model (Fixed batch=1, fixed 1x1x64x376 shape, no dynamic axes)
       │
       ▼
ONNX Runtime Execution Provider Priority:
  ├── 1. Qualcomm QNN HTP (Hexagon NPU / DSP)
  └── 2. CPU fallback (Local CPU Execution Provider)
       │
       ▼
Prediction & Uncertainty Triage (Confidence < 0.60 -> Maintenance Check Recommended)
```

---

## Installation

Ensure Python 3.10–3.14 is installed. Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Organize 16 kHz mono WAV recordings in the following directory layout:

```text
data/
├── normal/
│   └── *.wav
├── imbalance_rubbing/
│   └── *.wav
└── maintenance_check/
    └── *.wav
```

---

## Commands

### Training
```bash
python scripts/train.py
```

### ONNX Export
```bash
python scripts/export_onnx.py
```

### Quantization
```bash
python scripts/quantize.py
```

### Run Application
```bash
streamlit run app.py
```

---

## Qualcomm QNN HTP & CPU Fallback

- **Qualcomm Snapdragon NPU Acceleration**: The inference engine detects `QNNExecutionProvider` and attempts to load the Qualcomm Hexagon Tensor Processor backend (`QnnHtp.dll` or `libQnnHtp.so`).
- **CPU Fallback**: If the QNN execution provider or HTP library is not present on the host device, the pipeline automatically falls back to `CPUExecutionProvider` without crashing or reporting unverified NPU execution.

