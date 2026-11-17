"""Loading of the classification CSV and its reproducible data identity.

A checkpoint must be bound to the *actual samples* it was trained on.  The
identity is a SHA-256 over the parsed training tensors in file order:

    feature column names, shape, float32 X rows, int64 y rows

so it changes when any numeric value changes **or** when rows are reordered
(a pure permutation of the same rows is a different training sequence).  The
raw file bytes are hashed separately for diagnostics only; re-formatting a
CSV without changing the parsed samples does not change the identity.

The file is read exactly once per load: data and identity therefore always
describe the same bytes, and later edits on disk cannot alter the identity of
data that is already loaded in memory.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

LABEL_COLUMN = "label"
FINGERPRINT_ALGORITHM = "sha256:canonical-float32-rows-v1"
_FINGERPRINT_DOMAIN = b"resumable-training/dataset-identity/v1\0"


class DatasetLoadError(ValueError):
    """The dataset is missing, unreadable, or not a valid classification CSV."""


@dataclass(frozen=True)
class Dataset:
    """Parsed samples plus the identity of exactly those samples."""

    X: torch.Tensor
    y: torch.Tensor
    feature_cols: tuple[str, ...]
    path: str
    fingerprint: str
    file_sha256: str
    file_size: int
    num_samples: int
    num_features: int
    num_classes: int

    def info(self) -> dict:
        """Checkpoint-serialisable description of this dataset."""
        return {
            "path": self.path,
            "fingerprint": self.fingerprint,
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "file_sha256": self.file_sha256,
            "file_size": self.file_size,
            "num_samples": self.num_samples,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "feature_columns": list(self.feature_cols),
        }


def compute_dataset_fingerprint(X: torch.Tensor, y: torch.Tensor,
                                feature_cols) -> str:
    """Stable hash of the training tensors (values *and* row order).

    float32 / int64 little-endian bytes make the result independent of the
    host byte order and of CSV formatting such as ``1.0`` vs ``1.00``.
    """
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("dataset tensors have inconsistent shapes")
    h = hashlib.sha256()
    h.update(_FINGERPRINT_DOMAIN)
    h.update(json.dumps(list(feature_cols), ensure_ascii=True,
                        separators=(",", ":")).encode("ascii"))
    h.update(struct.pack(">QQ", int(X.shape[0]), int(X.shape[1])))
    xb = X.detach().cpu().contiguous().to(torch.float32).numpy()
    yb = y.detach().cpu().contiguous().to(torch.int64).numpy()
    h.update(xb.astype("<f4", copy=False).tobytes(order="C"))
    h.update(yb.astype("<i8", copy=False).tobytes(order="C"))
    return f"sha256:{h.hexdigest()}"


def load_classification_dataset(path: str) -> Dataset:
    """Read, parse and fingerprint the CSV in one pass over one byte snapshot.

    Raises :class:`DatasetLoadError` (a ``ValueError``) for a missing file,
    encoding problems, or malformed/empty data.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except FileNotFoundError as exc:
        raise DatasetLoadError(f"dataset file not found: {p}") from exc
    except OSError as exc:
        raise DatasetLoadError(f"dataset file cannot be read: {p}: {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(f"{p}: dataset is not valid UTF-8: {exc}") from exc

    xs: list[list[float]] = []
    ys: list[int] = []
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or LABEL_COLUMN not in reader.fieldnames:
            raise DatasetLoadError(
                f"{p}: CSV must contain a '{LABEL_COLUMN}' column")
        feature_cols = [c for c in reader.fieldnames if c != LABEL_COLUMN]
        if not feature_cols:
            raise DatasetLoadError(f"{p}: CSV has no feature columns")
        for line_no, row in enumerate(reader, start=2):
            try:
                xs.append([float(row[c]) for c in feature_cols])
                ys.append(int(row[LABEL_COLUMN]))
            except (TypeError, ValueError) as exc:
                raise DatasetLoadError(
                    f"{p}: invalid value at line {line_no}: {exc}") from exc
    except csv.Error as exc:
        raise DatasetLoadError(f"{p}: malformed CSV: {exc}") from exc

    if not xs:
        raise DatasetLoadError(f"{p}: dataset is empty")
    if len({len(row) for row in xs}) != 1:
        raise DatasetLoadError(f"{p}: rows have inconsistent feature counts")

    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    fingerprint = compute_dataset_fingerprint(X, y, feature_cols)
    return Dataset(
        X=X,
        y=y,
        feature_cols=tuple(feature_cols),
        path=str(p),
        fingerprint=fingerprint,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        file_size=len(raw),
        num_samples=int(X.shape[0]),
        num_features=int(X.shape[1]),
        num_classes=int(y.max().item()) + 1,
    )


def load_classification_csv(path: str):
    """Backwards-compatible helper: return ``(X, y, feature_cols)``."""
    ds = load_classification_dataset(path)
    return ds.X, ds.y, list(ds.feature_cols)
