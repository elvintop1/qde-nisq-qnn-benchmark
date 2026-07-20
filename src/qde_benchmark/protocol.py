from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_ENCODINGS = (
    "binary",
    "integer",
    "one_hot",
    "amplitude_histogram",
    "amplitude_mottonen",
    "amplitude_sparse",
    "angle",
    "dense_angle",
    "reupload_angle",
    "iqp",
    "hamiltonian",
    "trainable_kernel",
)

ENCODER_WIDTHS = {
    "binary": 16,
    "integer": 32,
    "one_hot": 64,
    "amplitude_histogram": 4,
    "amplitude_mottonen": 4,
    "amplitude_sparse": 4,
    "angle": 16,
    "dense_angle": 8,
    "reupload_angle": 16,
    "iqp": 16,
    "hamiltonian": 16,
    "trainable_kernel": 16,
}

ENCODER_PARAMETER_COUNTS = {
    name: (64 if name == "reupload_angle" else 16 if name == "trainable_kernel" else 0)
    for name in PAPER_ENCODINGS
}


@dataclass(frozen=True)
class PaperConfig:
    protocol: str
    datasets: dict[str, dict[str, str]]
    sample_size: int
    train_size: int
    validation_size: int
    test_size: int
    split_seeds: tuple[int, ...]
    pca_components: int
    qnn_layers: int
    epochs: int
    batch_size: int
    shots: int
    spsa_a: float
    spsa_c: float
    initialization_seed: int
    initialization_std: float
    classification_threshold: float
    transpiler_seed: int
    transpiler_optimization_level: int
    native_gates: tuple[str, ...]
    mps_min_width: int
    mps_bond_dimension: int
    mps_truncation_threshold: float
    resource_inputs_per_dataset_split: int
    encodings: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PaperConfig":
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("split_seeds", "native_gates", "encodings"):
            raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.encodings != PAPER_ENCODINGS:
            raise ValueError("The paper configuration must list the twelve encodings in protocol order.")
        if self.pca_components != 16:
            raise ValueError("The submitted paper protocol is defined at PCA k=16.")
        if self.train_size + self.validation_size + self.test_size != self.sample_size:
            raise ValueError("Train, validation, and test sizes must sum to sample_size.")
        if self.split_seeds != (1234, 1235, 1236):
            raise ValueError("The paper uses split seeds 1234, 1235, and 1236.")
        if self.qnn_layers != 2 or self.shots != 1024:
            raise ValueError("The paper uses two QNN layers and 1024 shots.")

    def qnn_parameter_count(self, width: int) -> int:
        return 2 * self.qnn_layers * int(width)

    def total_parameter_count(self, encoding: str) -> int:
        return ENCODER_PARAMETER_COUNTS[encoding] + self.qnn_parameter_count(
            ENCODER_WIDTHS[encoding]
        )

    def simulator_method(self, width: int) -> str:
        return "matrix_product_state" if int(width) >= self.mps_min_width else "statevector"

