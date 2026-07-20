from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import transpile

from .backend import build_fake_runtime
from .model import add_full_register_measurement, bind_model, build_qnn_model, probabilities_from_counts
from .preprocessing import DataSplit
from .protocol import PaperConfig


@dataclass(frozen=True)
class TrainedModel:
    encoding: str
    parameters: np.ndarray
    best_epoch: int
    validation_loss: float
    test_accuracy: float


def binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def train_encoding(encoding: str, split: DataSplit, config: PaperConfig) -> TrainedModel:
    """Train QNN parameters and, where present, encoding parameters jointly with SPSA."""
    template = build_qnn_model(encoding, split.X_train[0], layers=config.qnn_layers)
    parameter_count = len(template.parameters)
    rng = np.random.default_rng(config.initialization_seed)
    weights = config.initialization_std * rng.standard_normal(parameter_count)
    runtime = build_fake_runtime(template.circuit.num_qubits, config)

    best_weights = weights.copy()
    best_loss = float("inf")
    best_epoch = 0
    evaluation_index = 0
    for epoch in range(config.epochs):
        permutation = rng.permutation(len(split.X_train))
        for start in range(0, len(permutation), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            delta = rng.choice(np.asarray([-1.0, 1.0]), size=parameter_count)
            c_step = config.spsa_c / np.power(1.0 + epoch, 0.101)
            a_step = config.spsa_a / np.power(10.0 + epoch, 0.602)
            plus = _probability_one(
                encoding,
                split.X_train[indices],
                weights + c_step * delta,
                config,
                runtime,
                seed_offset=evaluation_index,
            )
            evaluation_index += 1
            minus = _probability_one(
                encoding,
                split.X_train[indices],
                weights - c_step * delta,
                config,
                runtime,
                seed_offset=evaluation_index,
            )
            evaluation_index += 1
            loss_plus = binary_cross_entropy(split.y_train[indices], plus)
            loss_minus = binary_cross_entropy(split.y_train[indices], minus)
            gradient = ((loss_plus - loss_minus) / (2.0 * c_step)) * delta
            weights = weights - a_step * gradient

        validation = _probability_one(
            encoding,
            split.X_validation,
            weights,
            config,
            runtime,
            seed_offset=evaluation_index,
        )
        evaluation_index += 1
        validation_loss = binary_cross_entropy(split.y_validation, validation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = weights.copy()
            best_epoch = epoch + 1

    test_probability = _probability_one(
        encoding,
        split.X_test,
        best_weights,
        config,
        runtime,
        seed_offset=evaluation_index,
    )
    predictions = (test_probability >= config.classification_threshold).astype(int)
    accuracy = float(np.mean(predictions == split.y_test))
    return TrainedModel(encoding, best_weights, best_epoch, best_loss, accuracy)


def _probability_one(encoding, features, weights, config, runtime, *, seed_offset: int):
    circuits = []
    for row in features:
        model = build_qnn_model(encoding, row, layers=config.qnn_layers)
        bound = bind_model(model, weights)
        circuits.append(add_full_register_measurement(bound))
    compiled = transpile(
        circuits,
        backend=runtime.transpile_backend,
        optimization_level=config.transpiler_optimization_level,
        seed_transpiler=config.transpiler_seed,
        num_processes=1,
    )
    result = runtime.simulator.run(
        compiled,
        shots=config.shots,
        seed_simulator=config.initialization_seed + int(seed_offset),
    ).result()
    values = []
    for index in range(len(compiled)):
        probabilities = probabilities_from_counts(result.get_counts(index))
        values.append(probabilities[1])
    return np.asarray(values, dtype=float)

