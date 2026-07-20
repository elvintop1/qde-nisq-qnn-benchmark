from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml, get_data_home


def load_binary_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the binary task named in the paper without distributing dataset files."""
    normalized = name.lower().strip()
    if normalized == "mnist":
        dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        labels = np.asarray(dataset.target).astype(str)
        mask = np.isin(labels, ["0", "1"])
        return np.asarray(dataset.data[mask], dtype=np.float32), labels[mask].astype(int)
    if normalized == "cifar10":
        return _load_cifar10_airplane_automobile()
    raise KeyError(f"Unsupported dataset: {name}")


def _load_cifar10_airplane_automobile() -> tuple[np.ndarray, np.ndarray]:
    """Load a checksum-pinned CIFAR-10 PNG mirror in original index order."""
    url = "https://data.pjreddie.com/files/cifar.tgz"
    expected = "3059b477333a52eb7c8f322706be56628646d7c8da6d86734b0e6598de63dbf5"
    cache_dir = Path(get_data_home()) / "cifar10_pjreddie"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "cifar.tgz"
    if not archive.exists() or _sha256(archive) != expected:
        urllib.request.urlretrieve(url, archive)
    if _sha256(archive) != expected:
        raise RuntimeError("The downloaded CIFAR-10 archive failed its SHA-256 check.")

    class_labels = {"airplane": 0, "automobile": 1}
    records: list[tuple[int, int, np.ndarray, int]] = []
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            stem = Path(member.name).stem
            if "_" not in stem:
                continue
            index_text, class_name = stem.split("_", 1)
            if class_name not in class_labels or not index_text.isdigit():
                continue
            stream = handle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Could not read {member.name}")
            with Image.open(stream) as image:
                row = np.asarray(image.convert("RGB"), dtype=np.float32)
            split_rank = 0 if "/train/" in member.name else 1
            records.append(
                (split_rank, int(index_text), row.transpose(2, 0, 1).reshape(-1), class_labels[class_name])
            )
    records.sort(key=lambda item: (item[0], item[1]))
    if len(records) != 12_000:
        raise RuntimeError(f"Expected 12,000 airplane/automobile images, found {len(records)}.")
    return np.vstack([item[2] for item in records]), np.asarray([item[3] for item in records])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

