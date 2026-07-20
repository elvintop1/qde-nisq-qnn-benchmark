from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile

from .backend import build_fake_runtime
from .encodings import build_encoding
from .model import bind_model, build_qnn_model
from .protocol import PaperConfig


@dataclass(frozen=True)
class ResourceRecord:
    encoding: str
    width: int
    encoding_depth: int
    encoding_ecr: int
    complete_depth: int
    complete_ecr: int
    encoding_parameters: int
    total_parameters: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_structural_resources(
    encoding: str,
    x: np.ndarray,
    config: PaperConfig,
    *,
    binding_seed: int = 1234,
) -> ResourceRecord:
    """Measure hardware-targeted compiled structure, not checkpoint-specific minima."""
    encoded = build_encoding(encoding, x)
    model = build_qnn_model(encoding, x, layers=config.qnn_layers)
    rng = np.random.default_rng(int(binding_seed))
    encoder_values = _nonzero_values(rng, len(encoded.trainable_parameters))
    model_values = _nonzero_values(rng, len(model.parameters))

    bound_encoding = encoded.circuit.assign_parameters(
        dict(zip(encoded.trainable_parameters, encoder_values)), inplace=False
    )
    bound_model = bind_model(model, model_values)
    runtime = build_fake_runtime(encoded.width, config)
    compiled_encoding, compiled_model = transpile(
        [bound_encoding, bound_model],
        backend=runtime.transpile_backend,
        optimization_level=config.transpiler_optimization_level,
        seed_transpiler=config.transpiler_seed,
        num_processes=1,
    )
    return ResourceRecord(
        encoding=encoding,
        width=encoded.width,
        encoding_depth=int(compiled_encoding.depth() or 0),
        encoding_ecr=_ecr_count(compiled_encoding),
        complete_depth=int(compiled_model.depth() or 0),
        complete_ecr=_ecr_count(compiled_model),
        encoding_parameters=len(encoded.trainable_parameters),
        total_parameters=len(model.parameters),
    )


def balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(int(seed))
    classes = np.unique(y)
    per_class = count // len(classes)
    chosen = [rng.choice(np.flatnonzero(y == label), size=per_class, replace=False) for label in classes]
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices


def _nonzero_values(rng: np.random.Generator, count: int) -> np.ndarray:
    if count == 0:
        return np.zeros(0, dtype=float)
    values = 0.05 * rng.standard_normal(int(count))
    values[np.isclose(values, 0.0)] = 1e-3
    return values


def _ecr_count(circuit: QuantumCircuit) -> int:
    counts = Counter({str(name): int(value) for name, value in circuit.count_ops().items()})
    return int(counts.get("ecr", 0))

