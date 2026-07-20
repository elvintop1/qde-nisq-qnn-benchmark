from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baselines import evaluate_classical_baselines
from .datasets import load_binary_dataset
from .preprocessing import build_pca_split, prepare_encoding_data
from .protocol import PAPER_ENCODINGS, PaperConfig
from .resources import balanced_indices, measure_structural_resources
from .training import train_encoding


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qde-benchmark",
        description="Run the source-only PCA-16/QNN protocol described in the manuscript.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "resources", "classical"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/paper.json"))
        command.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
        command.add_argument("--seed", type=int, default=1234)
        command.add_argument("--output", type=Path)

    train = subparsers.choices["train"]
    train.add_argument("--encoding", choices=PAPER_ENCODINGS, required=True)

    resources = subparsers.choices["resources"]
    resources.add_argument("--encoding", choices=PAPER_ENCODINGS, action="append")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PaperConfig.load(args.config)
    if args.seed not in config.split_seeds:
        raise ValueError(f"seed must be one of {config.split_seeds}")
    X, y = load_binary_dataset(args.dataset)
    split = build_pca_split(
        X,
        y,
        seed=args.seed,
        sample_size=config.sample_size,
        pca_components=config.pca_components,
    )

    if args.command == "train":
        prepared = prepare_encoding_data(split, args.encoding)
        record = train_encoding(args.encoding, prepared, config)
        payload: dict[str, Any] = {
            "protocol": config.protocol,
            "dataset": args.dataset,
            "seed": args.seed,
            "encoding": record.encoding,
            "best_epoch": record.best_epoch,
            "validation_loss": record.validation_loss,
            "test_accuracy": record.test_accuracy,
            "parameters": record.parameters.tolist(),
        }
        default_output = Path("outputs") / f"train_{args.dataset}_{args.encoding}_{args.seed}.json"
    elif args.command == "resources":
        names = tuple(args.encoding or PAPER_ENCODINGS)
        rows: list[dict[str, object]] = []
        for encoding in names:
            prepared = prepare_encoding_data(split, encoding)
            indices = balanced_indices(
                prepared.y_train,
                config.resource_inputs_per_dataset_split,
                args.seed,
            )
            for input_index in indices:
                record = measure_structural_resources(
                    encoding,
                    prepared.X_train[int(input_index)],
                    config,
                    binding_seed=args.seed + int(input_index),
                )
                rows.append(
                    {
                        "protocol": config.protocol,
                        "dataset": args.dataset,
                        "seed": args.seed,
                        "input_index": int(input_index),
                        **record.as_dict(),
                    }
                )
        payload = {"records": rows}
        default_output = Path("outputs") / f"resources_{args.dataset}_{args.seed}.json"
    else:
        rows = [record.as_dict() for record in evaluate_classical_baselines(split, seed=args.seed)]
        payload = {
            "protocol": config.protocol,
            "dataset": args.dataset,
            "seed": args.seed,
            "records": rows,
        }
        default_output = Path("outputs") / f"classical_{args.dataset}_{args.seed}.json"

    output = args.output or default_output
    _write_json(output, payload)
    print(f"Wrote {output}")
    return 0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
