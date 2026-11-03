import sys
from pathlib import Path
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import MotorConditionCNN
from config import (
    N_MELS,
    SAMPLE_RATE,
    CLIP_DURATION,
    HOP_LENGTH,
    MODEL_PATH,
    MODEL_DIR,
)


def main():

    model = MotorConditionCNN(
        num_classes=3
    )

    pth_path = MODEL_DIR / "motor_condition.pth"
    if pth_path.exists():
        model.load_state_dict(
            torch.load(
                str(pth_path),
                map_location="cpu",
            )
        )
        print(f"Loaded checkpoint from {pth_path}")
    else:
        print(
            f"[INFO] Checkpoint {pth_path} not found. "
            "Exporting baseline architecture for edge pipeline readiness."
        )

    model.eval()

    time_frames = (
        int(
            (SAMPLE_RATE * CLIP_DURATION)
            / HOP_LENGTH
        ) + 1
    )

    dummy_input = torch.randn(
        1,
        1,
        N_MELS,
        time_frames,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_PATH),
        input_names=[
            "audio_features"
        ],
        output_names=[
            "class_logits"
        ],
        opset_version=17,
        dynamic_axes=None,
    )

    print(
        f"ONNX model exported: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
