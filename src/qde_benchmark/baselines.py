from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import DataSplit


@dataclass(frozen=True)
class BaselineRecord:
    model: str
    validation_accuracy: float
    test_accuracy: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_classical_baselines(split: DataSplit, *, seed: int) -> list[BaselineRecord]:
    """Evaluate the three PCA-16 reference models reported beside the QNN study."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "rbf_svm": SVC(kernel="rbf", random_state=seed),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=2000,
            random_state=seed,
        ),
    }
    records: list[BaselineRecord] = []
    for name, estimator in models.items():
        pipeline = make_pipeline(StandardScaler(), estimator)
        pipeline.fit(split.X_train, split.y_train)
        records.append(
            BaselineRecord(
                model=name,
                validation_accuracy=float(
                    np.mean(pipeline.predict(split.X_validation) == split.y_validation)
                ),
                test_accuracy=float(np.mean(pipeline.predict(split.X_test) == split.y_test)),
            )
        )
    return records
