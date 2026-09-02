from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inferix_model_foundations.sample import generate_text, load_model
from inferix_model_foundations.train import TrainConfig, pick_device, train


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    return tmp_path / "m0-demo"


def test_inspect_forward_shapes() -> None:
    from inferix_model_foundations.config import GPTConfig
    from inferix_model_foundations.data import get_batch, load_dataset
    from inferix_model_foundations.model import GPT

    device = pick_device()
    ds = load_dataset()
    config = GPTConfig(vocab_size=ds.vocab_size)
    model = GPT(config).to(device)
    x, y = get_batch(ds.train_ids, config.block_size, batch_size=4, device=device)
    logits, loss = model(x, y)
    assert logits.shape == (4, config.block_size, ds.vocab_size)
    assert loss.ndim == 0


def test_train_dry_run_writes_checkpoint(ckpt_dir: Path) -> None:
    ckpt_path = ckpt_dir / "ckpt.pt"
    summary = train(
        train_cfg=TrainConfig.from_env(dry_run=True),
        ckpt_path=ckpt_path,
        log_path=ckpt_dir / "loss.csv",
        manifest_path=ckpt_dir / "manifest.json",
    )
    assert ckpt_path.exists()
    assert summary["train_mode"] == "dry_run"
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state" in payload
    assert payload["train_mode"] == "dry_run"


def test_sample_after_reload(ckpt_dir: Path) -> None:
    ckpt_path = ckpt_dir / "ckpt.pt"
    train(
        train_cfg=TrainConfig.from_env(dry_run=True),
        ckpt_path=ckpt_path,
        log_path=ckpt_dir / "loss.csv",
        manifest_path=ckpt_dir / "manifest.json",
    )
    text = generate_text(ckpt_path, prompt="\n", tokens=20, temperature=1.0)
    assert isinstance(text, str)
    assert len(text) > 1

    device = pick_device()
    model, _stoi, _itos, payload = load_model(ckpt_path, device)
    assert model.param_count() > 0
    assert payload["step"] >= 1


def test_manifest_labels_train_mode(ckpt_dir: Path) -> None:
    manifest_path = ckpt_dir / "manifest.json"
    train(
        train_cfg=TrainConfig.from_env(dry_run=True),
        ckpt_path=ckpt_dir / "ckpt.pt",
        log_path=ckpt_dir / "loss.csv",
        manifest_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["phase"] == "M0a"
    assert manifest["train_mode"] == "dry_run"
    assert "honest_label" in manifest
