"""CLI entry for m0b-inspect console script."""

from __future__ import annotations

import argparse
import json

from inferix_model_foundations.open_weights import (
    DEFAULT_MODEL_ID,
    format_report,
    inspect_open_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M0b — inspect tiny HF open weights")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--no-forward", action="store_true", help="skip forward pass")
    args = parser.parse_args()

    report = inspect_open_weights(args.model_id, run_forward=not args.no_forward)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
