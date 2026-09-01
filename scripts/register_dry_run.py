"""Dry-run registration against model-registry (Era B prep, optional)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from inferix_model_foundations.paths import CKPT_PATH, MANIFEST_PATH, REPO_ROOT


def register_dry_run(*, ckpt_path: Path = CKPT_PATH) -> dict[str, object]:
    registry_cli = REPO_ROOT.parent / "model-registry" / "model_registry" / "cli.py"
    if not registry_cli.exists():
        return {
            "status": "skipped",
            "reason": "model-registry not found at ../model-registry",
        }

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")

    cmd = [
        sys.executable,
        "-m",
        "model_registry.cli",
        "--dry-run",
        "--model-name",
        "m0-foundations-nanogpt",
        "--run-id",
        "m0-demo",
        "--git-sha",
        "0000000000000000000000000000000000000000",
        "--dataset-hash",
        "m0-tinyshakespeare-char",
        "--artifact-prefix",
        "m0-foundations/m0-demo/",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT.parent / "model-registry",
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "checkpoint": str(ckpt_path),
        "manifest": json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else None,
        "note": "dry-run only — not a RouteIQ model_id; M0 is literacy, not production",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Register M0 checkpoint via model-registry --dry-run")
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    args = parser.parse_args()
    result = register_dry_run(ckpt_path=args.ckpt)
    print(json.dumps(result, indent=2))
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
