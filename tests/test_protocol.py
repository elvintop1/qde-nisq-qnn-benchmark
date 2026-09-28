from pathlib import Path

import numpy as np

from qde_benchmark.encodings import build_encoding
from qde_benchmark.model import build_qnn_model, class_from_bitstring, probabilities_from_counts
from qde_benchmark.protocol import (
    ENCODER_PARAMETER_COUNTS,
    ENCODER_WIDTHS,
    PAPER_ENCODINGS,
    PaperConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration_and_capacity_rules() -> None:
    config = PaperConfig.load(ROOT / "configs" / "paper.json")
    assert config.encodings == PAPER_ENCODINGS
    assert config.simulator_method(16) == "statevector"
    assert config.simulator_method(32) == "matrix_product_state"
    assert config.simulator_method(64) == "matrix_product_state"
    assert config.total_parameter_count("reupload_angle") == 128
    assert config.total_parameter_count("trainable_kernel") == 80


def test_all_encoding_widths_and_parameter_counts() -> None:
    x = np.linspace(0.05, 0.95, 16)
    x /= np.linalg.norm(x)
    for name in PAPER_ENCODINGS:
        encoding = build_encoding(name, x)
        assert encoding.width == ENCODER_WIDTHS[name]
        assert len(encoding.trainable_parameters) == ENCODER_PARAMETER_COUNTS[name]
        model = build_qnn_model(name, x, layers=2)
        assert len(model.parameters) == ENCODER_PARAMETER_COUNTS[name] + 4 * ENCODER_WIDTHS[name]


def test_amplitude_variants_share_the_same_circuit_loader() -> None:
    x = np.linspace(0.05, 0.95, 16)
    x /= np.linalg.norm(x)
    names = ("amplitude_histogram", "amplitude_mottonen", "amplitude_sparse")

    for name in names:
        encoding = build_encoding(name, x)
        assert encoding.circuit.count_ops() == {"state_preparation": 1}


def test_binary_decoder_is_least_significant_bit_marginal() -> None:
    assert class_from_bitstring("0000") == 0
    assert class_from_bitstring("1110") == 0
    assert class_from_bitstring("0001") == 1
    assert class_from_bitstring("1011") == 1
    probabilities = probabilities_from_counts({"0000": 100, "1110": 300, "0001": 200, "1011": 400})
    np.testing.assert_allclose(probabilities, np.asarray([0.4, 0.6]))
