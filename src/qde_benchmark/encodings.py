from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import StatePreparation, UCRYGate

from .protocol import ENCODER_PARAMETER_COUNTS, ENCODER_WIDTHS, PAPER_ENCODINGS


@dataclass(frozen=True)
class EncodingCircuit:
    name: str
    circuit: QuantumCircuit
    trainable_parameters: tuple[Parameter, ...]

    @property
    def width(self) -> int:
        return self.circuit.num_qubits


def build_encoding(name: str, x: np.ndarray) -> EncodingCircuit:
    values = np.asarray(x, dtype=float).ravel()
    if len(values) != 16:
        raise ValueError(f"The paper protocol expects 16 retained features, got {len(values)}.")
    builders = {
        "binary": _binary,
        "integer": _integer,
        "one_hot": _one_hot,
        "amplitude_histogram": _amplitude_histogram,
        "amplitude_mottonen": _amplitude_mottonen,
        "amplitude_sparse": _amplitude_sparse,
        "angle": _angle,
        "dense_angle": _dense_angle,
        "reupload_angle": _reupload,
        "iqp": _iqp,
        "hamiltonian": _hamiltonian,
        "trainable_kernel": _trainable_kernel,
    }
    if name not in builders:
        raise KeyError(f"Unknown encoding {name!r}; expected one of {PAPER_ENCODINGS}.")
    built = builders[name](name, values)
    expected_width = ENCODER_WIDTHS[name]
    expected_parameters = ENCODER_PARAMETER_COUNTS[name]
    if built.width != expected_width:
        raise AssertionError(f"{name}: expected width {expected_width}, got {built.width}.")
    if len(built.trainable_parameters) != expected_parameters:
        raise AssertionError(
            f"{name}: expected {expected_parameters} encoding parameters, "
            f"got {len(built.trainable_parameters)}."
        )
    return built


def _binary(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(16, name=name)
    for qubit, bit in enumerate(x >= 0.5):
        if bit:
            circuit.x(qubit)
    return EncodingCircuit(name, circuit, ())


def _integer(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(32, name=name)
    quantized = np.clip(np.round(3.0 * x), 0, 3).astype(int)
    for feature, value in enumerate(quantized):
        for bit in range(2):
            if (int(value) >> bit) & 1:
                circuit.x(2 * feature + bit)
    return EncodingCircuit(name, circuit, ())


def _one_hot(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(64, name=name)
    bins = np.clip(np.floor(4.0 * x), 0, 3).astype(int)
    for feature, active in enumerate(bins):
        circuit.x(4 * feature + int(active))
    return EncodingCircuit(name, circuit, ())


def _amplitude_histogram(name: str, x: np.ndarray) -> EncodingCircuit:
    """Load amplitudes produced by histogram preprocessing."""
    return _amplitude_state_preparation(name, x)


def _amplitude_mottonen(name: str, x: np.ndarray) -> EncodingCircuit:
    """Prepare a normalized real vector with Möttönen uniformly controlled rotations."""
    amplitudes = np.asarray(x, dtype=float)
    norm = float(np.linalg.norm(amplitudes))
    if norm <= 1e-15:
        raise ValueError("Möttönen state preparation is undefined for an all-zero vector.")
    amplitudes = amplitudes / norm

    width = int(math.log2(len(amplitudes)))
    circuit = QuantumCircuit(width, name=name)
    for target in reversed(range(width)):
        angles = _mottonen_y_angles(amplitudes, target)
        controls = list(range(target + 1, width))
        if controls:
            circuit.append(UCRYGate(angles), [target, *controls])
        else:
            circuit.ry(angles[0], target)
    return EncodingCircuit(name, circuit, ())


def _amplitude_sparse(name: str, x: np.ndarray) -> EncodingCircuit:
    """Load amplitudes produced by top-k sparse preprocessing."""
    return _amplitude_state_preparation(name, x)


def _amplitude_state_preparation(name: str, x: np.ndarray) -> EncodingCircuit:
    """State-preparation stage shared by the histogram and sparse variants."""
    width = int(math.log2(len(x)))
    circuit = QuantumCircuit(width, name=name)
    circuit.append(StatePreparation(np.asarray(x, dtype=complex), normalize=True), circuit.qubits)
    return EncodingCircuit(name, circuit, ())


def _mottonen_y_angles(amplitudes: np.ndarray, target: int) -> list[float]:
    """Return the uniformly controlled RY angles for one Möttönen cascade level."""
    block_size = 1 << (target + 1)
    half = 1 << target
    angles: list[float] = []
    for start in range(0, len(amplitudes), block_size):
        block = amplitudes[start : start + block_size]
        if target == 0:
            angles.append(2.0 * math.atan2(float(block[1]), float(block[0])))
            continue
        zero_norm = float(np.linalg.norm(block[:half]))
        one_norm = float(np.linalg.norm(block[half:]))
        angles.append(2.0 * math.atan2(one_norm, zero_norm))
    return angles


def _angle(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(16, name=name)
    for qubit, value in enumerate(x):
        circuit.ry(float(value), qubit)
    return EncodingCircuit(name, circuit, ())


def _dense_angle(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(8, name=name)
    for qubit in range(8):
        circuit.ry(float(x[2 * qubit]), qubit)
        circuit.rz(float(x[2 * qubit + 1]), qubit)
    return EncodingCircuit(name, circuit, ())


def _reupload(name: str, x: np.ndarray) -> EncodingCircuit:
    parameters = ParameterVector("phi", 64)
    circuit = QuantumCircuit(16, name=name)
    for layer in range(4):
        for qubit, value in enumerate(x):
            circuit.ry(float(value) + parameters[16 * layer + qubit], qubit)
        _append_ring(circuit)
    return EncodingCircuit(name, circuit, tuple(parameters))


def _iqp(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(16, name=name)
    circuit.h(range(16))
    for qubit, value in enumerate(x):
        circuit.rz(2.0 * float(value), qubit)
    _append_all_pairs(circuit, x, scale=2.0)
    return EncodingCircuit(name, circuit, ())


def _hamiltonian(name: str, x: np.ndarray) -> EncodingCircuit:
    circuit = QuantumCircuit(16, name=name)
    circuit.h(range(16))
    for qubit, value in enumerate(x):
        circuit.rz(float(value), qubit)
    _append_all_pairs(circuit, x, scale=1.0)
    for qubit in range(16):
        circuit.rx(np.pi / 4.0, qubit)
    return EncodingCircuit(name, circuit, ())


def _trainable_kernel(name: str, x: np.ndarray) -> EncodingCircuit:
    parameters = ParameterVector("vartheta", 16)
    circuit = QuantumCircuit(16, name=name)
    circuit.h(range(16))
    for qubit, value in enumerate(x):
        circuit.rz(2.0 * parameters[qubit] * float(value), qubit)
    _append_all_pairs(circuit, x, scale=2.0)
    return EncodingCircuit(name, circuit, tuple(parameters))


def _append_ring(circuit: QuantumCircuit) -> None:
    width = circuit.num_qubits
    for qubit in range(width - 1):
        circuit.cx(qubit, qubit + 1)
    if width > 2:
        circuit.cx(width - 1, 0)


def _append_all_pairs(circuit: QuantumCircuit, x: np.ndarray, *, scale: float) -> None:
    for left in range(16):
        for right in range(left + 1, 16):
            circuit.cx(left, right)
            circuit.rz(scale * float(x[left] * x[right]), right)
            circuit.cx(left, right)
