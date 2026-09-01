"""M0b — inspect a tiny Hugging Face GPT-2 for architecture literacy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_MODEL_ID = "sshleifer/tiny-gpt2"


@dataclass
class LayerShape:
    name: str
    shape: tuple[int, ...]
    params: int


@dataclass
class OpenWeightsReport:
    model_id: str
    vocab_size: int
    special_tokens: dict[str, str | None]
    n_layer: int
    n_head: int
    n_embd: int
    max_position_embeddings: int
    param_count: int
    trainable_param_count: int
    memory_fp32_mb: float
    memory_fp16_mb: float
    major_modules: list[str]
    layer_shapes: list[LayerShape]
    forward_ok: bool
    forward_logits_shape: tuple[int, ...] | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer_shapes"] = [asdict(ls) for ls in self.layer_shapes]
        return data


def _memory_mb(param_count: int, bytes_per_param: int) -> float:
    return round(param_count * bytes_per_param / (1024**2), 2)


def inspect_open_weights(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    run_forward: bool = True,
) -> OpenWeightsReport:
    """Load a tiny HF causal LM and return an architecture dissection report."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    config = model.config

    special_tokens = {
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
    }

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_shapes: list[LayerShape] = []
    for name, param in model.named_parameters():
        layer_shapes.append(LayerShape(name=name, shape=tuple(param.shape), params=param.numel()))

    forward_ok = False
    forward_logits_shape: tuple[int, ...] | None = None
    if run_forward:
        import torch

        text = "Hello, open weights."
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        forward_ok = True
        forward_logits_shape = tuple(outputs.logits.shape)

    notes = [
        "BPE tokenizer with byte-level pretokenization (GPT-2 family)",
        "Pre-norm is absent — GPT-2 uses post-norm inside blocks",
        "Weight tying: wte shares weights with lm_head (like nanoGPT)",
        "KV cache used at inference via past_key_values (not in nanoGPT toy)",
    ]

    max_pos = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", 1024)

    return OpenWeightsReport(
        model_id=model_id,
        vocab_size=int(config.vocab_size),
        special_tokens=special_tokens,
        n_layer=int(config.n_layer),
        n_head=int(config.n_head),
        n_embd=int(config.n_embd),
        max_position_embeddings=int(max_pos),
        param_count=param_count,
        trainable_param_count=trainable,
        memory_fp32_mb=_memory_mb(param_count, 4),
        memory_fp16_mb=_memory_mb(param_count, 2),
        major_modules=[name for name, _ in model.named_children()],
        layer_shapes=layer_shapes,
        forward_ok=forward_ok,
        forward_logits_shape=forward_logits_shape,
        notes=notes,
    )


def format_report(report: OpenWeightsReport) -> str:
    lines = [
        "=== M0b open-weight inspect ===",
        f"model_id              {report.model_id}",
        f"vocab_size            {report.vocab_size:,}",
        f"n_layer / n_head      {report.n_layer} / {report.n_head}",
        f"n_embd                {report.n_embd}",
        f"context (n_pos)      {report.max_position_embeddings}",
        f"param_count           {report.param_count:,}",
        f"memory (fp32)         {report.memory_fp32_mb} MB",
        f"memory (fp16)         {report.memory_fp16_mb} MB",
        f"special_tokens        {report.special_tokens}",
        f"major_modules         {report.major_modules}",
        "",
        "top parameter tensors:",
    ]
    for ls in report.layer_shapes[:12]:
        lines.append(f"  {ls.name:40s} {tuple(ls.shape)!s:20s} {ls.params:,}")
    if len(report.layer_shapes) > 12:
        lines.append(f"  ... {len(report.layer_shapes) - 12} more tensors")
    if report.forward_logits_shape:
        lines.append(f"\nforward logits shape  {report.forward_logits_shape}")
    lines.append("ok")
    return "\n".join(lines)
