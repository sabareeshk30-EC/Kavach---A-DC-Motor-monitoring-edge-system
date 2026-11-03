import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from preprocess import preprocess_audio
from config import BASELINE_PATH


def create_baseline(audio_files: List[str | Path]):
    features = []

    for path in audio_files:
        feature = preprocess_audio(path)
        feature = feature.flatten()
        features.append(feature)

    if not features:
        raise ValueError("Cannot create baseline: No valid audio files provided.")

    features = np.asarray(features)
    baseline = np.median(
        features,
        axis=0,
    )

    BASELINE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        BASELINE_PATH,
        "w",
    ) as f:
        json.dump(
            {
                "feature": baseline.tolist(),
                "count": len(features),
            },
            f,
            indent=2,
        )

    return baseline


def load_baseline() -> Optional[np.ndarray]:
    if not BASELINE_PATH.exists():
        return None
    try:
        with open(BASELINE_PATH, "r") as f:
            data = json.load(f)
            return np.asarray(data["feature"], dtype=np.float32)
    except Exception:
        return None


def baseline_distance(
    feature,
    baseline,
) -> float:
    feature = np.asarray(feature).flatten()
    baseline = np.asarray(baseline).flatten()

    numerator = np.linalg.norm(
        feature - baseline
    )

    denominator = (
        np.linalg.norm(baseline)
        + 1e-8
    )

    return float(
        numerator / denominator
    )
