# -------------------------------------------------------------
# scripts/train.py  —  Kavach EdgeInspect v2
# -------------------------------------------------------------
"""
Improved training script for MotorConditionCNN v2.

Improvements over v1
--------------------
- Reads pre-split data/train/ and data/validation/ (produced by
  prepare_dataset.py) — no duplicate splitting logic.
- Class-weighted CrossEntropyLoss for imbalanced datasets.
- CosineAnnealingLR scheduler for smooth LR decay.
- Early stopping (patience = 5 epochs) — saves best val-acc model.
- Per-epoch console table with loss + accuracy for both splits.
- Saves training history to artifacts/training_history.json.
- Supports CUDA / MPS / CPU auto-detection.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import librosa

# ----------------------------------------------------------------
# Project root
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import MotorConditionCNN
from preprocess import audio_to_logmel
from config import SAMPLE_RATE, CLIP_DURATION

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
MODELS_DIR    = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

CLASS_MAP = {
    "normal":            0,
    "imbalance_rubbing": 1,
    "maintenance_check": 2,
}

EPOCHS     = 40
BATCH_SIZE = 16
LR         = 3e-4
PATIENCE   = 7   # early-stopping patience


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------
class MotorDataset(Dataset):
    def __init__(self, file_list: list[tuple[str, int]]):
        self.samples = []
        target = int(SAMPLE_RATE * CLIP_DURATION)
        for path, label in file_list:
            try:
                audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
            except Exception:
                audio = np.zeros(target, dtype=np.float32)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            else:
                audio = audio[:target]
            feature = audio_to_logmel(audio.astype(np.float32))
            tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            self.samples.append((tensor, torch.tensor(label, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def collect_files(split: str) -> list[tuple[str, int]]:
    """Gather (path, label_id) pairs from data/<split>/<class_folder>/."""
    split_dir = DATA_DIR / split
    files: list[tuple[str, int]] = []
    for folder, label in CLASS_MAP.items():
        class_dir = split_dir / folder
        if not class_dir.exists():
            continue
        for wav in sorted(class_dir.glob("*.wav")):
            files.append((str(wav), label))
    return files


def class_weights(file_list: list[tuple[str, int]], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(CLASS_MAP), dtype=np.float32)
    for _, label in file_list:
        counts[label] += 1
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(CLASS_MAP)   # normalise
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ----------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train() if training else model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            loss = criterion(out, y)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += (out.argmax(dim=1) == y).sum().item()
            total      += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("  Kavach EdgeInspect — Training  (MotorConditionCNN v2)")
    print("=" * 65)

    # Collect files
    train_files = collect_files("train")
    val_files   = collect_files("validation")

    if not train_files:
        # Fallback: look in data/ root (legacy layout)
        print("[INFO] data/train/ empty – falling back to data/ root layout.")
        for folder, label in CLASS_MAP.items():
            for wav in (DATA_DIR / folder).glob("*.wav"):
                train_files.append((str(wav), label))

    if not train_files:
        raise RuntimeError(
            "No WAV files found. Run `python scripts/prepare_dataset.py` first."
        )

    print(f"\n  Train samples : {len(train_files)}")
    print(f"  Val samples   : {len(val_files)}")

    # Class distribution
    from collections import Counter
    dist = Counter(lbl for _, lbl in train_files)
    id_to_name = {v: k for k, v in CLASS_MAP.items()}
    for lbl_id, cnt in sorted(dist.items()):
        print(f"    class {lbl_id} ({id_to_name[lbl_id]}): {cnt}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # Data loaders
    np.random.shuffle(train_files)
    train_ds = MotorDataset(train_files)
    val_ds   = MotorDataset(val_files) if val_files else MotorDataset(train_files[-max(1, len(train_files)//5):])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = MotorConditionCNN(num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}\n")

    # Loss, optimiser, scheduler
    cw        = class_weights(train_files, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training state
    best_val_acc  = 0.0
    patience_ctr  = 0
    history: list[dict] = []

    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>8}  {'ValAcc':>8}  {'LR':>10}")
    print("  " + "-" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        marker = "  *" if vl_acc > best_val_acc else ""
        print(
            f"  {epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.3%}  "
            f"{vl_loss:>8.4f}  {vl_acc:>8.3%}  {lr_now:>10.2e}{marker}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_acc":  round(tr_acc,  5),
            "val_loss":   round(vl_loss, 5),
            "val_acc":    round(vl_acc,  5),
            "lr":         round(lr_now,  8),
        })

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODELS_DIR / "motor_condition_best.pth")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch} (patience={PATIENCE}).")
                break

    # Always save final checkpoint too
    torch.save(model.state_dict(), MODELS_DIR / "motor_condition.pth")

    # Persist history
    hist_path = ARTIFACTS_DIR / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": round(best_val_acc, 5), "history": history}, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Best val accuracy : {best_val_acc:.2%}")
    print(f"  Best checkpoint   : {MODELS_DIR / 'motor_condition_best.pth'}")
    print(f"  Training history  : {hist_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
