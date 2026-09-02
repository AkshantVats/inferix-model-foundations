"""Train the tiny GPT and write a real checkpoint under checkpoints/m0-demo/."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from inferix_model_foundations.config import GPTConfig, TrainConfig
from inferix_model_foundations.data import get_batch, load_dataset
from inferix_model_foundations.model import GPT
from inferix_model_foundations.paths import CKPT_PATH, LOG_PATH, MANIFEST_PATH


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    eval_iters: int = 20,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, split in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = float(losses.mean())
    model.train()
    return out


def train(
  *,
  train_cfg: TrainConfig | None = None,
  ckpt_path: Path = CKPT_PATH,
  log_path: Path = LOG_PATH,
  manifest_path: Path = MANIFEST_PATH,
) -> dict[str, object]:
    train_cfg = train_cfg or TrainConfig.from_env()
    device = pick_device()
    ds = load_dataset()
    config = GPTConfig(vocab_size=ds.vocab_size)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "elapsed_s"])

    print(f"train_mode={train_cfg.train_mode}")
    print(f"device={device}  params={model.param_count():,}  vocab={ds.vocab_size}")
    print(f"train chars={len(ds.train_ids):,}  val chars={len(ds.val_ids):,}")
    print(f"writing {ckpt_path}")

    t0 = time.time()
    best_val = float("inf")
    best_step = 0
    model.train()
    for step in range(1, train_cfg.max_steps + 1):
        x, y = get_batch(ds.train_ids, config.block_size, train_cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % train_cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"step {step:4d}  train_loss={loss.item():.4f}  elapsed={elapsed:.1f}s")

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            stats = estimate_loss(
                model,
                ds.train_ids,
                ds.val_ids,
                config.block_size,
                train_cfg.batch_size,
                device,
                eval_iters=train_cfg.eval_iters,
            )
            elapsed = time.time() - t0
            print(
                f"eval  step {step:4d}  train={stats['train']:.4f}  val={stats['val']:.4f}"
            )
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{stats['train']:.6f}", f"{stats['val']:.6f}", f"{elapsed:.1f}"]
                )
            if stats["val"] < best_val:
                best_val = stats["val"]
                best_step = step
                payload = {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "stoi": ds.stoi,
                    "itos": ds.itos,
                    "step": step,
                    "val_loss": best_val,
                    "train_mode": train_cfg.train_mode,
                    "device": str(device),
                }
                torch.save(payload, ckpt_path)
                print(f"saved checkpoint val_loss={best_val:.4f}")

    manifest = {
        "phase": "M0a",
        "pack": "foundations",
        "train_mode": train_cfg.train_mode,
        "honest_label": (
            "real from-scratch train"
            if train_cfg.train_mode == "from_scratch"
            else f"labeled {train_cfg.train_mode} — fewer steps, still real backprop"
        ),
        "checkpoint": str(ckpt_path.relative_to(ckpt_path.parent.parent.parent)),
        "best_step": best_step,
        "best_val_loss": best_val,
        "device": str(device),
        "params": model.param_count(),
        "train_config": asdict(train_cfg),
        "master_plan": "../master-plan-for-agents/models/foundations/BUILD_PLAN.md",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "train_mode": train_cfg.train_mode,
        "best_val_loss": best_val,
        "best_step": best_step,
        "ckpt_path": str(ckpt_path),
        "manifest_path": str(manifest_path),
        "device": str(device),
    }
    print(f"done. best val_loss={best_val:.4f}  ckpt={ckpt_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M0a nanoGPT checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pytest/CI path: 5 steps, tiny batch, still real backprop",
    )
    args = parser.parse_args()
    train_cfg = TrainConfig.from_env(dry_run=args.dry_run)
    train(train_cfg=train_cfg)


if __name__ == "__main__":
    main()
