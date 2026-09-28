import numpy as np

from qde_benchmark.preprocessing import DataSplit, build_pca_split, prepare_encoding_data


def test_split_sizes_and_train_fitted_transforms() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6000, 24))
    y = np.repeat(np.asarray([0, 1]), 3000)
    split = build_pca_split(X, y, seed=1234, sample_size=5000, pca_components=16)
    assert split.X_train.shape == (3000, 16)
    assert split.X_validation.shape == (1000, 16)
    assert split.X_test.shape == (1000, 16)
    prepared = prepare_encoding_data(split, "angle")
    assert np.min(prepared.X_train) >= 0.0
    assert np.max(prepared.X_train) <= np.pi + 1e-12


def test_amplitude_rows_are_normalized() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(6000, 20))
    y = np.repeat(np.asarray([0, 1]), 3000)
    split = build_pca_split(X, y, seed=1234, sample_size=5000, pca_components=16)
    for name in ("amplitude_histogram", "amplitude_mottonen", "amplitude_sparse"):
        prepared = prepare_encoding_data(split, name)
        np.testing.assert_allclose(np.linalg.norm(prepared.X_train, axis=1), 1.0, atol=1e-10)


def test_three_amplitude_preprocessing_rules_are_distinct() -> None:
    row = np.asarray([[-4.0, 3.0, -2.0, 1.0] + [0.1] * 12])
    labels = np.asarray([0])
    split = DataSplit(row, labels, row, labels, row, labels, seed=1234)

    histogram = prepare_encoding_data(split, "amplitude_histogram").X_train[0]
    mottonen = prepare_encoding_data(split, "amplitude_mottonen").X_train[0]
    sparse = prepare_encoding_data(split, "amplitude_sparse").X_train[0]

    assert np.all(histogram >= 0.0)
    assert mottonen[0] < 0.0
    assert np.count_nonzero(mottonen) == 16
    assert np.count_nonzero(sparse) == 4
    assert not np.allclose(histogram, mottonen)
    assert not np.allclose(mottonen, sparse)
