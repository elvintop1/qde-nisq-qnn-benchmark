from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qde_benchmark import training
from qde_benchmark.preprocessing import DataSplit
from qde_benchmark.protocol import PaperConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("size", "drop_last", "expected_batches", "expected_examples"),
    [
        (3000, True, 187, 2992),
        (3000, False, 188, 3000),
        (32, True, 2, 32),
        (32, False, 2, 32),
        (16, True, 1, 16),
        (8, True, 0, 0),
        (8, False, 1, 8),
    ],
)
def test_minibatches_preserve_shuffled_indices(
    size: int, drop_last: bool, expected_batches: int, expected_examples: int
) -> None:
    permutation = np.random.default_rng(7).permutation(size)
    batches = list(training._minibatches(permutation, 16, drop_last=drop_last))
    assert len(batches) == expected_batches
    assert sum(map(len, batches)) == expected_examples
    if drop_last:
        assert all(len(batch) == 16 for batch in batches)
    if batches:
        np.testing.assert_array_equal(
            np.concatenate(batches), permutation[:expected_examples]
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_minibatches_reject_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(training._minibatches(np.arange(32), batch_size, drop_last=True))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_configuration_rejects_nonpositive_batch_size(batch_size: int) -> None:
    config = PaperConfig.load(ROOT / "configs" / "paper.json")
    with pytest.raises(ValueError, match="batch_size must be positive"):
        replace(config, batch_size=batch_size).validate()


@pytest.mark.parametrize("drop_last", ["true", 1, None])
def test_configuration_requires_boolean_drop_last(drop_last: object) -> None:
    config = PaperConfig.load(ROOT / "configs" / "paper.json")
    with pytest.raises(ValueError, match="drop_last must be a boolean"):
        replace(config, drop_last=drop_last).validate()


def test_configuration_requires_a_complete_batch_when_dropping() -> None:
    config = PaperConfig.load(ROOT / "configs" / "paper.json")
    with pytest.raises(ValueError, match="at least one complete training batch"):
        replace(config, batch_size=config.train_size + 1).validate()


def _split(size: int) -> DataSplit:
    return DataSplit(
        X_train=np.arange(size, dtype=float).reshape(-1, 1),
        y_train=np.zeros(size, dtype=int),
        X_validation=np.full((1000, 1), -1.0),
        y_validation=np.ones(1000, dtype=int),
        X_test=np.full((1000, 1), -2.0),
        y_test=np.ones(1000, dtype=int),
        seed=1234,
    )


@pytest.mark.parametrize("drop_last,updates_per_epoch", [(True, 187), (False, 188)])
def test_training_loop_counts_updates_without_running_quantum_circuits(
    monkeypatch: pytest.MonkeyPatch, drop_last: bool, updates_per_epoch: int
) -> None:
    config = replace(
        PaperConfig.load(ROOT / "configs" / "paper.json"), drop_last=drop_last
    )
    split = _split(3000)
    calls = []

    def probability_one(encoding, features, weights, config, runtime, *, seed_offset):
        if features is split.X_validation:
            phase = "validation"
        elif features is split.X_test:
            phase = "test"
        else:
            phase = "training"
        calls.append((phase, len(features), seed_offset))
        return np.full(len(features), 0.5)

    monkeypatch.setattr(
        training,
        "build_qnn_model",
        lambda *args, **kwargs: SimpleNamespace(
            parameters=("angle",), circuit=SimpleNamespace(num_qubits=4)
        ),
    )
    monkeypatch.setattr(training, "build_fake_runtime", lambda *args: object())
    monkeypatch.setattr(training, "_probability_one", probability_one)

    record = training.train_encoding("angle", split, config)

    training_calls = [call for call in calls if call[0] == "training"]
    assert len(training_calls) == 2 * config.epochs * updates_per_epoch
    batch_sizes = [size for _, size, _ in training_calls[::2]]
    if drop_last:
        assert batch_sizes == [16] * 1870
    else:
        assert batch_sizes == ([16] * 187 + [8]) * config.epochs
    assert [size for phase, size, _ in calls if phase == "validation"] == [1000] * 10
    assert [size for phase, size, _ in calls if phase == "test"] == [1000]
    assert calls[-1][0] == "test"
    assert [offset for _, _, offset in calls] == list(range(len(calls)))
    assert record.best_epoch == 1


def test_training_rejects_an_empty_complete_batch_before_building_circuits() -> None:
    config = PaperConfig.load(ROOT / "configs" / "paper.json")
    with pytest.raises(ValueError, match="at least one complete training batch"):
        training.train_encoding("angle", _split(8), config)


def test_test_evaluation_uses_lowest_validation_loss_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(PaperConfig.load(ROOT / "configs" / "paper.json"), epochs=3)
    split = _split(17)
    validation_weights = []
    test_weights = []
    validation_probabilities = iter([0.6, 0.9, 0.7])

    def probability_one(encoding, features, weights, config, runtime, *, seed_offset):
        if features is split.X_validation:
            validation_weights.append(weights.copy())
            return np.full(len(features), next(validation_probabilities))
        if features is split.X_test:
            test_weights.append(weights.copy())
            return np.full(len(features), 0.8)
        return np.full(len(features), 0.5 + 0.1 * np.tanh(weights[0]))

    monkeypatch.setattr(
        training,
        "build_qnn_model",
        lambda *args, **kwargs: SimpleNamespace(
            parameters=("angle",), circuit=SimpleNamespace(num_qubits=4)
        ),
    )
    monkeypatch.setattr(training, "build_fake_runtime", lambda *args: object())
    monkeypatch.setattr(training, "_probability_one", probability_one)

    record = training.train_encoding("angle", split, config)

    assert record.best_epoch == 2
    assert len(validation_weights) == 3
    assert len(test_weights) == 1
    assert not np.array_equal(validation_weights[1], validation_weights[2])
    np.testing.assert_array_equal(test_weights[0], validation_weights[1])
