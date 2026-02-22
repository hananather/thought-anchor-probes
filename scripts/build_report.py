#!/usr/bin/env python
"""Build a markdown report from saved metrics and predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import ensure_parent_dirs, load_config
from ta_probe.report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument("--top-examples", type=int, default=5, help="Examples per model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    report = build_report(
        metrics_path=config.paths.metrics_json,
        predictions_path=config.paths.predictions_parquet,
        output_path=config.paths.report_md,
        top_n_examples=args.top_examples,
    )
    print(report)


if __name__ == "__main__":
    main()
