from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ANGLE_ENCODINGS = {"angle", "dense_angle", "reupload_angle", "iqp", "hamiltonian", "trainable_kernel"}
AMPLITUDE_ENCODINGS = {"amplitude_histogram", "amplitude_mottonen", "amplitude_sparse"}
DISCRETE_ENCODINGS = {"binary", "integer", "one_hot"}


@dataclass(frozen=True)
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    seed: int


def build_pca_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    sample_size: int = 5000,
    pca_components: int = 16,
) -> DataSplit:
    X_sample, _, y_sample, _ = train_test_split(
        np.asarray(X),
        np.asarray(y),
        train_size=sample_size,
        stratify=y,
        random_state=seed,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_sample,
        y_sample,
        test_size=0.4,
        stratify=y_sample,
        random_state=seed,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=seed,
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    validation_scaled = scaler.transform(X_validation)
    test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=pca_components, random_state=seed)
    return DataSplit(
        X_train=pca.fit_transform(train_scaled),
        y_train=y_train.astype(int),
        X_validation=pca.transform(validation_scaled),
        y_validation=y_validation.astype(int),
        X_test=pca.transform(test_scaled),
        y_test=y_test.astype(int),
        seed=seed,
    )


def prepare_encoding_data(split: DataSplit, encoding: str) -> DataSplit:
    matrices = (split.X_train, split.X_validation, split.X_test)
    if encoding in ANGLE_ENCODINGS:
        transformed = _fit_minmax(matrices, (0.0, np.pi))
    elif encoding in DISCRETE_ENCODINGS:
        transformed = _fit_minmax(matrices, (0.0, 1.0))
    elif encoding in AMPLITUDE_ENCODINGS:
        transformed = tuple(_prepare_amplitude(matrix, encoding) for matrix in matrices)
    else:
        raise KeyError(f"Unknown encoding: {encoding}")
    return DataSplit(
        transformed[0],
        split.y_train,
        transformed[1],
        split.y_validation,
        transformed[2],
        split.y_test,
        split.seed,
    )


def _fit_minmax(
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray], feature_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(matrices[0]), scaler.transform(matrices[1]), scaler.transform(matrices[2])


def _prepare_amplitude(matrix: np.ndarray, encoding: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    if encoding == "amplitude_histogram":
        values = np.abs(values)
        values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
        values = np.sqrt(values)
    elif encoding == "amplitude_sparse":
        keep = min(4, values.shape[1])
        indices = np.argpartition(np.abs(values), -keep, axis=1)[:, -keep:]
        mask = np.zeros_like(values, dtype=bool)
        mask[np.arange(len(values))[:, None], indices] = True
        values[~mask] = 0.0
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0.0, 1.0, norms)

