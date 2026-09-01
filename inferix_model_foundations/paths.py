from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "m0-demo"
CKPT_PATH = CHECKPOINT_DIR / "ckpt.pt"
LOG_PATH = CHECKPOINT_DIR / "loss.csv"
MANIFEST_PATH = CHECKPOINT_DIR / "manifest.json"
