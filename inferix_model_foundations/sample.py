"""Generate text from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from inferix_model_foundations.config import GPTConfig
from inferix_model_foundations.model import GPT
from inferix_model_foundations.paths import CKPT_PATH
from inferix_model_foundations.train import pick_device


def load_model(
    ckpt_path: Path, device: torch.device
) -> tuple[GPT, dict[str, int], dict[int, str], dict[str, object]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**payload["config"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    itos = {int(k): v for k, v in payload["itos"].items()}
    return model, payload["stoi"], itos, payload


def generate_text(
    ckpt_path: Path,
    prompt: str = "\n",
    tokens: int = 120,
    temperature: float = 0.8,
) -> str:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path} — run m0-train first")

    device = pick_device()
    model, stoi, itos, _payload = load_model(ckpt_path, device)
    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        raise ValueError(f"prompt has chars not in vocab: {unknown!r}")

    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=tokens, temperature=temperature)
    return "".join(itos[int(i)] for i in out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    text = generate_text(
        args.ckpt,
        prompt=args.prompt,
        tokens=args.tokens,
        temperature=args.temperature,
    )
    print(text)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(
        f"\n--- reloaded {args.ckpt}  step={payload['step']}  "
        f"train_mode={payload.get('train_mode', 'unknown')} ---"
    )


if __name__ == "__main__":
    main()
