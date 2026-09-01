"""Day 1 exit: data loads, batch shapes print, untrained forward works."""

from __future__ import annotations

import math

from inferix_model_foundations.config import GPTConfig
from inferix_model_foundations.data import get_batch, load_dataset
from inferix_model_foundations.model import GPT
from inferix_model_foundations.train import pick_device


def main() -> None:
    device = pick_device()
    ds = load_dataset()
    config = GPTConfig(vocab_size=ds.vocab_size)
    model = GPT(config).to(device)

    x, y = get_batch(ds.train_ids, config.block_size, batch_size=4, device=device)
    logits, loss = model(x, y)

    print("=== M0a inspect ===")
    print(f"device          {device}")
    print(f"corpus chars    {len(ds.text):,}")
    print(f"vocab_size      {ds.vocab_size}")
    print(f"train / val     {len(ds.train_ids):,} / {len(ds.val_ids):,}")
    print(f"x batch shape   {tuple(x.shape)}")
    print(f"y batch shape   {tuple(y.shape)}")
    print(f"logits shape    {tuple(logits.shape)}")
    print(f"loss (random)   {loss.item():.4f}   # ~ln(vocab)≈{math.log(ds.vocab_size):.2f}")
    print(f"params          {model.param_count():,}")
    print(f"sample decode   {ds.decode(x[0, :40].tolist())!r}")
    print("ok")


if __name__ == "__main__":
    main()
