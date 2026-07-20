# QDE under NISQ resource constraints: QNN case-study code

This source-only repository provides a reference implementation of the protocol
described in **“Quantum Data Encoding under NISQ Resource Constraints: A QNN
Case Study.”** It contains no experimental results, trained checkpoints, raw
metrics, or paper tables.

The code evaluates twelve quantum data encodings in a shared PCA-16/QNN
pipeline. It implements:

- MNIST zero-versus-one and CIFAR-10 airplane-versus-automobile tasks;
- three stratified splits with seeds 1234–1236;
- training-fitted standardization and PCA with `k = 16`;
- fixed and trainable computational-basis, local-rotation,
  amplitude-preparation, and correlation-oriented encodings;
- a shared two-layer `Ry`/`Rz` QNN with a logical CX ring;
- 1024-shot full-register computational-basis sampling;
- the fixed binary decoder `g(b) = b0`;
- joint SPSA optimization of QNN and trainable-encoding parameters;
- FakeSherbrooke-derived noise and hardware-targeted transpilation; and
- separate encoding-only and complete-circuit resource measurements.

## Important interpretation boundary

The resource scripts quantify **structural, hardware-targeted compilation cost**
under the fixed transpilation setting. They do not estimate live-device runtime,
fidelity, energy, or the minimum circuit cost attainable through
checkpoint-specific parameter binding and algebraic simplification.

The original lab execution environment is not present in this repository. This
release therefore documents and tests the manuscript protocol as a reference
implementation; it does not claim line-for-line identity with an unavailable
private lab checkout.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The main quantum dependencies are pinned to the versions recorded by the local
experiment artifact: Qiskit 2.3.1, Qiskit Aer 0.17.2, and Qiskit IBM Runtime
0.47.0.

## Protocol audit

Run the lightweight conformance tests before launching experiments:

```bash
pytest
python scripts/audit_release.py
```

The audit checks the twelve encoding names, qubit widths, trainable parameter
counts, QNN parameter rule, readout definition, MPS boundary, and the absence of
committed result artifacts.

## Running experiments

The commands below write only to `outputs/`, which is ignored by Git.

```bash
qde-benchmark train --config configs/paper.json --dataset mnist --encoding angle
qde-benchmark resources --config configs/paper.json --dataset mnist
```

Full noisy training is computationally expensive. In particular, the 32- and
64-qubit basis configurations use matrix-product-state simulation, while widths
up to 16 use statevector simulation.

## Repository policy

Do not commit generated CSV files, metrics, plots, logs, trained parameters, or
dataset caches. Run `python scripts/audit_release.py` before every push.

## License

MIT License. See [LICENSE](LICENSE).

