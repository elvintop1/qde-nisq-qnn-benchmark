from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from .protocol import PaperConfig


@dataclass(frozen=True)
class FakeRuntime:
    transpile_backend: GenericBackendV2
    simulator: AerSimulator
    noise_model: NoiseModel
    selected_physical_qubits: tuple[int, ...]


def build_fake_runtime(width: int, config: PaperConfig) -> FakeRuntime:
    """Build the deterministic connected FakeSherbrooke execution envelope."""
    return _build_fake_runtime_cached(
        int(width),
        config.native_gates,
        config.initialization_seed,
        config.mps_min_width,
        config.mps_bond_dimension,
        config.mps_truncation_threshold,
    )


@lru_cache(maxsize=16)
def _build_fake_runtime_cached(
    width: int,
    native_gates: tuple[str, ...],
    initialization_seed: int,
    mps_min_width: int,
    mps_bond_dimension: int,
    mps_truncation_threshold: float,
) -> FakeRuntime:
    snapshot = FakeSherbrooke()
    selected = _connected_prefix(snapshot, int(width))
    physical_to_local = {physical: local for local, physical in enumerate(selected)}
    snapshot_edges = [(int(left), int(right)) for left, right in snapshot.coupling_map.get_edges()]
    reduced_edges = [
        [physical_to_local[left], physical_to_local[right]]
        for left, right in snapshot_edges
        if left in physical_to_local and right in physical_to_local
    ]
    available = set(snapshot.configuration().basis_gates)
    basis_gates = [gate for gate in native_gates if gate in available]
    transpile_backend = GenericBackendV2(
        int(width),
        basis_gates=basis_gates,
        coupling_map=CouplingMap(reduced_edges),
        seed=initialization_seed,
        noise_info=False,
    )

    full_noise = NoiseModel.from_backend(snapshot)
    reduced_errors = []
    for error in full_noise.to_dict().get("errors", []):
        remapped = [
            [physical_to_local[int(qubit)] for qubit in qubits]
            for qubits in error.get("gate_qubits", [])
            if all(int(qubit) in physical_to_local for qubit in qubits)
        ]
        if remapped:
            reduced = copy.deepcopy(error)
            reduced["gate_qubits"] = remapped
            reduced_errors.append(reduced)
    noise_model = NoiseModel.from_dict({"errors": reduced_errors})
    simulator = AerSimulator(noise_model=noise_model)
    method = "matrix_product_state" if width >= mps_min_width else "statevector"
    options: dict[str, object] = {
        "method": method,
        "shot_branching_enable": True,
        "shot_branching_sampling_enable": True,
    }
    if method == "matrix_product_state":
        options.update(
            {
                "matrix_product_state_max_bond_dimension": mps_bond_dimension,
                "matrix_product_state_truncation_threshold": mps_truncation_threshold,
            }
        )
    simulator.set_options(**options)
    return FakeRuntime(transpile_backend, simulator, noise_model, selected)


def _connected_prefix(snapshot: FakeSherbrooke, width: int) -> tuple[int, ...]:
    adjacency = {qubit: set() for qubit in range(snapshot.num_qubits)}
    for left, right in snapshot.coupling_map.get_edges():
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
    selected: list[int] = []
    frontier = [0]
    seen = {0}
    while frontier and len(selected) < width:
        node = frontier.pop(0)
        selected.append(node)
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    if len(selected) != width:
        raise RuntimeError(f"Could not select {width} connected FakeSherbrooke qubits.")
    return tuple(selected)
