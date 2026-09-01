# inferix-model-foundations

**M0a** implementation for the [Inferix Agent Ecosystem](https://github.com/AkshantVats/inferix) ORC program — Karpathy-style nanoGPT trained from scratch on a Mac (CPU or MPS).

| Field | Value |
|-------|-------|
| Master plan | [`../master-plan-for-agents/MASTER_PLAN.md`](../master-plan-for-agents/MASTER_PLAN.md) |
| Build plan | [`../master-plan-for-agents/models/foundations/BUILD_PLAN.md`](../master-plan-for-agents/models/foundations/BUILD_PLAN.md) |
| Overview | [`../master-plan-for-agents/models/foundations/OVERVIEW.md`](../master-plan-for-agents/models/foundations/OVERVIEW.md) |
| Quality bar | [`../master-plan-for-agents/QUALITY_BAR.md`](../master-plan-for-agents/QUALITY_BAR.md) |
| Phase | **M0a** (nanoGPT) — M0b open dissect still pending |
| Era | **A only** — no Inferix gate, not a RouteIQ `model_id` |

## What this is

A **real** character-level GPT loop at toy scale:

```
text → tokens → embeddings → attention + MLP → logits → loss → backprop → checkpoint → generate
```

Outputs land in `checkpoints/m0-demo/` (`ckpt.pt`, `loss.csv`, `manifest.json`).

## Setup (8 GB Mac friendly)

```bash
cd inferix-model-foundations
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run

```bash
# Day 1 — data + shapes + untrained forward
m0-inspect

# Day 3 — full train (~2 min MPS / <10 min CPU); writes checkpoints/m0-demo/
m0-train

# Day 4 — reload checkpoint and sample
m0-sample --prompt "ROMEO:" --tokens 200

# pytest smoke (dry-run: 5 steps, real backprop)
pytest -v
```

## Train modes (honest labels)

| Mode | How | Label |
|------|-----|-------|
| **from_scratch** (default) | 2000 steps, full tiny Shakespeare | `train_mode=from_scratch` in checkpoint + manifest |
| **simulate** | `FINETUNE_SIMULATE=1 m0-train` — 30 steps, still real backprop | `train_mode=simulate` |
| **dry_run** | `m0-train --dry-run` or pytest — 5 steps | `train_mode=dry_run` |

All modes write **real PyTorch checkpoints** with gradients applied — no stub weights.

## Optional: model-registry dry-run

After training, exercise the FineForge registry client without MinIO:

```bash
pip install -e '../model-registry[dev]'
python scripts/register_dry_run.py
```

This is Era B prep only. M0 is literacy, not production registration.

## Layout

```
inferix_model_foundations/
  config.py      GPT + train configs
  data.py        Tiny Shakespeare loader
  model.py       transformer (read this)
  train.py       training loop + manifest
  sample.py      reload + generate
  inspect.py     Day 1 smoke
checkpoints/m0-demo/   artifact output (gitignored)
tests/                 pytest smoke
```

## vs competitor ~50%

Reference: [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT).

| Matches | Deferred |
|---------|----------|
| Train / checkpoint / reload / sample | BPE tokenizer, multi-GPU, wandb |
| Loss CSV + manifest | FineForge UI, RouteIQ, DriftWatch |
| CPU/MPS local | Cluster training |

## Related repos

- Planning pack: `master-plan-for-agents/models/foundations/`
- Earlier prototype: `owned-llms/` (superseded by this ORC repo for M0a tracking)
