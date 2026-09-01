"""Character-level dataset loader for Tiny Shakespeare."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
import torch

from inferix_model_foundations.paths import DATA_DIR

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = DATA_DIR / "tinyshakespeare.txt"

# Tiny inline fallback when offline — enough chars for smoke tests.
_INLINE_CORPUS = (
    "ROMEO:\nBut soft! What light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n"
) * 40


def download_corpus(path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        response = requests.get(SHAKESPEARE_URL, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    except requests.RequestException:
        path.write_text(_INLINE_CORPUS, encoding="utf-8")
    return path


@dataclass
class CharDataset:
    text: str
    stoi: dict[str, int]
    itos: dict[int, str]
    data: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_dataset(path: Path | None = None, val_frac: float = 0.1) -> CharDataset:
    path = download_corpus(path or DATA_PATH)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * (1.0 - val_frac))
    return CharDataset(
        text=text,
        stoi=stoi,
        itos=itos,
        data=ids,
        train_ids=ids[:n],
        val_ids=ids[n:],
    )


def get_batch(
    split: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(split) - block_size, (batch_size,))
    x = torch.stack([split[i : i + block_size] for i in ix])
    y = torch.stack([split[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
