import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(
        "release_audit", ROOT / "scripts" / "audit_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracked = []
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "tracked_paths", lambda: list(tracked))
    return module, tmp_path, tracked


def _fixture_file(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture\n")
    return path


def test_release_audit_accepts_only_the_explicit_log_exception(audit_module) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, "training_logs/qnn_training_history.csv"))
    assert module.ALLOWED_LOG_FILES == {Path("training_logs/qnn_training_history.csv")}
    assert module.main() == 0


@pytest.mark.parametrize(
    "relative",
    [
        "log_loss.csv",
        "training_logs/log_loss.csv",
        "training_logs/other.csv",
        "training_logs/nested/log_loss.csv",
        "results/log_loss.csv",
    ],
)
def test_release_audit_rejects_other_result_files(audit_module, relative: str) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, relative))
    assert module.main() == 1


def test_log_exception_still_obeys_file_size_limit(
    audit_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, "training_logs/qnn_training_history.csv"))
    monkeypatch.setattr(module, "MAX_SOURCE_SIZE", 1)
    assert module.main() == 1


def test_published_log_identifies_test_accuracy_explicitly() -> None:
    path = ROOT / "training_logs" / "qnn_training_history.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert "test_accuracy" in reader.fieldnames
        assert "eval_accuracy" not in reader.fieldnames
        rows = list(reader)
    assert len(rows) == 720
    assert all(0 <= float(row["test_accuracy"]) <= 1 for row in rows)
