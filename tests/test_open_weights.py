from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from inferix_model_foundations.open_weights import (
    DEFAULT_MODEL_ID,
    OpenWeightsReport,
    format_report,
    inspect_open_weights,
)


def _fake_config() -> MagicMock:
    cfg = MagicMock()
    cfg.vocab_size = 50257
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 64
    cfg.n_pos = 1024
    return cfg


def _fake_model() -> torch.nn.Module:
    class FakeGPT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _fake_config()
            self.wte = torch.nn.Embedding(50257, 64)
            self.lm_head = torch.nn.Linear(64, 50257, bias=False)

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> MagicMock:
            batch, seq = input_ids.shape
            return MagicMock(logits=torch.zeros(batch, seq, 50257))

    return FakeGPT()


def _fake_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.bos_token = "<|endoftext|>"
    tok.eos_token = "<|endoftext|>"
    tok.pad_token = None
    tok.unk_token = None
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    return tok


@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_inspect_open_weights_mocked(mock_tok_cls: MagicMock, mock_model_cls: MagicMock) -> None:
    mock_tok_cls.return_value = _fake_tokenizer()
    mock_model_cls.return_value = _fake_model()

    report = inspect_open_weights(DEFAULT_MODEL_ID, run_forward=True)

    assert isinstance(report, OpenWeightsReport)
    assert report.model_id == DEFAULT_MODEL_ID
    assert report.vocab_size == 50257
    assert report.n_layer == 2
    assert report.param_count > 0
    assert report.forward_ok is True
    assert report.forward_logits_shape == (1, 5, 50257)
    assert report.memory_fp32_mb > 0
    assert "wte" in report.major_modules


def test_format_report_includes_key_fields() -> None:
    report = OpenWeightsReport(
        model_id="test/tiny",
        vocab_size=100,
        special_tokens={"bos_token": None, "eos_token": None, "pad_token": None, "unk_token": None},
        n_layer=2,
        n_head=2,
        n_embd=64,
        max_position_embeddings=128,
        param_count=1_000_000,
        trainable_param_count=1_000_000,
        memory_fp32_mb=3.81,
        memory_fp16_mb=1.91,
        major_modules=["transformer"],
        layer_shapes=[],
        forward_ok=True,
        forward_logits_shape=(1, 4, 100),
        notes=[],
    )
    text = format_report(report)
    assert "test/tiny" in text
    assert "param_count" in text
    assert "ok" in text
