from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from .encodings import EncodingCircuit, build_encoding


@dataclass(frozen=True)
class ModelCircuit:
    encoding: str
    circuit: QuantumCircuit
    encoding_parameters: tuple[Parameter, ...]
    qnn_parameters: tuple[Parameter, ...]

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return self.encoding_parameters + self.qnn_parameters


def build_qnn_model(encoding: str, x: np.ndarray, *, layers: int = 2) -> ModelCircuit:
    encoded: EncodingCircuit = build_encoding(encoding, x)
    width = encoded.width
    qnn_parameters = ParameterVector("theta", 2 * layers * width)
    circuit = QuantumCircuit(width, name=f"qnn_{encoding}")
    circuit.compose(encoded.circuit, inplace=True)
    cursor = 0
    for _ in range(layers):
        for qubit in range(width):
            circuit.ry(qnn_parameters[cursor], qubit)
            circuit.rz(qnn_parameters[cursor + 1], qubit)
            cursor += 2
        _append_ring(circuit)
    return ModelCircuit(
        encoding=encoding,
        circuit=circuit,
        encoding_parameters=encoded.trainable_parameters,
        qnn_parameters=tuple(qnn_parameters),
    )


def bind_model(model: ModelCircuit, values: np.ndarray) -> QuantumCircuit:
    numeric = np.asarray(values, dtype=float).ravel()
    if len(numeric) != len(model.parameters):
        raise ValueError(f"Expected {len(model.parameters)} parameters, got {len(numeric)}.")
    return model.circuit.assign_parameters(dict(zip(model.parameters, numeric)), inplace=False)


def add_full_register_measurement(circuit: QuantumCircuit) -> QuantumCircuit:
    measured = circuit.copy()
    measured.measure_all()
    return measured


def class_from_bitstring(bitstring: str) -> int:
    """Implement g(b)=b0 for Qiskit's displayed little-endian count strings."""
    compact = bitstring.replace(" ", "")
    if not compact or any(bit not in "01" for bit in compact):
        raise ValueError(f"Invalid measured bitstring: {bitstring!r}")
    return int(compact[-1])


def probabilities_from_counts(counts: dict[str, int]) -> np.ndarray:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("Counts must contain at least one shot.")
    class_counts = np.zeros(2, dtype=float)
    for bitstring, count in counts.items():
        class_counts[class_from_bitstring(bitstring)] += int(count)
    return class_counts / float(total)


def _append_ring(circuit: QuantumCircuit) -> None:
    width = circuit.num_qubits
    for qubit in range(width - 1):
        circuit.cx(qubit, qubit + 1)
    if width > 2:
        circuit.cx(width - 1, 0)

