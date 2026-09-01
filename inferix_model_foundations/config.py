from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 64
    log_every: int = 50
    eval_every: int = 250
    eval_iters: int = 20
    lr: float = 3e-3
    train_mode: str = "from_scratch"

    @classmethod
    def from_env(cls, *, dry_run: bool = False) -> TrainConfig:
        simulate = os.environ.get("FINETUNE_SIMULATE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if dry_run:
            return cls(
                max_steps=5,
                batch_size=8,
                log_every=1,
                eval_every=5,
                eval_iters=2,
                train_mode="dry_run",
            )
        if simulate:
            return cls(
                max_steps=30,
                batch_size=16,
                log_every=10,
                eval_every=30,
                eval_iters=3,
                train_mode="simulate",
            )
        return cls(train_mode="from_scratch")

    def with_overrides(self, **kwargs: object) -> TrainConfig:
        return replace(self, **kwargs)
