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


def test_release_audit_accepts_learning_curve_image(audit_module) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, "training_logs/qnn_learning_curves.png"))
    assert module.main() == 0


@pytest.mark.parametrize(
    "relative",
    [
        "log_loss.csv",
        "training_logs/log_loss.csv",
        "training_logs/qnn_training_history.csv",
        "training_logs/other.csv",
        "training_logs/nested/log_loss.csv",
        "results/log_loss.csv",
    ],
)
def test_release_audit_rejects_all_csv_result_files(audit_module, relative: str) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, relative))
    assert module.main() == 1


def test_learning_curve_image_obeys_file_size_limit(
    audit_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, root, tracked = audit_module
    tracked.append(_fixture_file(root, "training_logs/qnn_learning_curves.png"))
    monkeypatch.setattr(module, "MAX_SOURCE_SIZE", 1)
    assert module.main() == 1


def test_published_training_materials_include_only_the_learning_curve_image() -> None:
    paths = {path.name for path in (ROOT / "training_logs").iterdir() if path.is_file()}
    assert paths == {"qnn_learning_curves.png"}
