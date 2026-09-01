# M0b — Open-weight architecture notes

**Specimen:** [`sshleifer/tiny-gpt2`](https://huggingface.co/sshleifer/tiny-gpt2) (~0.4 MB fp32)  
**Inspect:** `m0b-inspect` or `python scripts/inspect_open_weights.py`  
**Date:** 2026-09-02

## Summary

| Metric | Value |
|--------|-------|
| `model_id` | `sshleifer/tiny-gpt2` |
| `vocab_size` | 50,257 (BPE) |
| `n_layer` | 2 |
| `n_head` | 2 |
| `n_embd` | 2 |
| `max_position_embeddings` | 1,024 |
| `param_count` | 102,714 |
| `memory_fp32` | 0.39 MB |
| `memory_fp16` | 0.20 MB |
| Forward logits shape | `(batch, seq, vocab)` → e.g. `(1, 5, 50257)` |

### Special tokens

| Token | Value |
|-------|-------|
| `bos_token` | `<|endoftext|>` |
| `eos_token` | `<|endoftext|>` |
| `pad_token` | *(none — left-pad in HF)* |
| `unk_token` | `<|endoftext|>` |

### Major modules

- `transformer` — `wte`, `wpe`, `h[0..1]` (attn + MLP blocks), `ln_f`
- `lm_head` — tied to `transformer.wte.weight`

### Representative layer shapes

| Tensor | Shape | Params |
|--------|-------|--------|
| `transformer.wte.weight` | (50257, 2) | 100,514 |
| `transformer.wpe.weight` | (1024, 2) | 2,048 |
| `transformer.h.0.attn.c_attn.weight` | (2, 6) | 12 |
| `transformer.h.0.mlp.c_fc.weight` | (2, 8) | 16 |

---

## nanoGPT vs open-weight GPT-2 (M0a vs M0b)

| Topic | nanoGPT (M0a) | Open weight (M0b) |
|-------|---------------|-------------------|
| **Tokenizer** | Character-level, vocab=65 | BPE byte-pair, vocab=50,257 |
| **Block / layer count** | 4 layers, 4 heads, embd=128 | 2 layers, 2 heads, embd=2 |
| **Attention** | Manual causal mask buffer; QKV via single `c_attn` linear | Same idea; fused QKV in `c_attn`; optional KV cache at inference |
| **Context length** | `block_size=64` (hard cap in forward) | `max_position_embeddings=1024` via `wpe` |
| **How you sample** | `model.generate()` — multinomial on last logit | HF `generate()` or manual; supports `past_key_values` |
| **What “your weights” means** | Checkpoint you trained (`checkpoints/m0-demo/ckpt.pt`, 809,856 params) | Weights you downloaded; later LoRA adapters in M1 |
| **Norm placement** | Pre-norm (`ln` before attn/MLP) | Post-norm (GPT-2 classic) |
| **Weight tying** | `wte` ↔ `lm_head` | Same pattern |
| **Memory (fp32)** | ~3.1 MB (809K params) | ~0.39 MB (103K params — toy HF model) |

### Same ideas, productionized in open weights

1. **Autoregressive next-token prediction** — CE loss on shifted labels in both.
2. **Transformer blocks** — attention + MLP residual stack.
3. **Embeddings** — token (`wte`) + position (`wpe` in GPT-2; nanoGPT also has `wpe`).
4. **Checkpoint semantics** — reload weights → same outputs (M0a Day 4 exit).

### Production additions (not in nanoGPT toy)

- BPE tokenizer + special tokens
- KV cache (`past_key_values`) for O(n) inference
- Hugging Face `config.json` + `safetensors`/`bin` artifact layout
- Scale: real 1B–3B instruct models reuse this skeleton with larger dims

---

## Next: M1 Era A

Instruction-tuning scaffold lives in [`../inferix-model-general-llm/`](../inferix-model-general-llm/).  
It can warm-start from the M0a nanoGPT checkpoint or run in `FINETUNE_SIMULATE` mode on Darwin without CUDA.
