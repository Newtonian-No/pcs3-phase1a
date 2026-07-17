"""Raw nine-channel UCI HAR preparation with subject-safe validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .temporal_logic import _sha256_file, _write_json_atomic, _write_npz_atomic


UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)
_SPLITS = ("train", "val", "test")


def _safe_extract_zip(archive: zipfile.ZipFile, destination: str | Path) -> list[str]:
    """Extract regular ZIP members only after validating every destination."""

    destination = Path(destination)
    destination_resolved = destination.resolve()
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    for info in archive.infolist():
        member_path = (destination / info.filename).resolve()
        try:
            member_path.relative_to(destination_resolved)
        except ValueError as exc:
            raise ValueError(f"unsafe ZIP member path: {info.filename}") from exc
        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(f"unsafe ZIP symbolic link: {info.filename}")
        validated.append((info, member_path))

    destination.mkdir(parents=True, exist_ok=True)
    for info, member_path in validated:
        if info.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue
        member_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, member_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return [info.filename for info, _ in validated]


def _find_dataset_dir(root: Path) -> Path:
    candidates = []
    for inertial in root.rglob("Inertial Signals"):
        if inertial.is_dir() and inertial.parent.name == "train":
            candidate = inertial.parent.parent
            if (candidate / "test" / "Inertial Signals").is_dir():
                candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"UCI HAR extracted layout not found below {root}")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _download_archive(path: Path, url: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def download_uci_har(
    root: str | Path,
    *,
    download_url: str = UCI_HAR_URL,
) -> dict[str, object]:
    """Download and securely extract the official archive with provenance."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "human_activity_recognition_using_smartphones.zip"
    if not archive_path.exists():
        _download_archive(archive_path, download_url)

    extracted = root / "extracted"
    outer_members: list[str] = []
    nested_members: dict[str, list[str]] = {}
    if not extracted.exists():
        temporary = root / ".extracting"
        if temporary.exists():
            shutil.rmtree(temporary)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                outer_members = _safe_extract_zip(archive, temporary)
            try:
                _find_dataset_dir(temporary)
            except FileNotFoundError:
                for nested_path in sorted(temporary.rglob("*.zip")):
                    nested_destination = nested_path.parent / nested_path.stem
                    with zipfile.ZipFile(nested_path) as nested_archive:
                        nested_members[str(nested_path.relative_to(temporary))] = _safe_extract_zip(
                            nested_archive,
                            nested_destination,
                        )
                    try:
                        _find_dataset_dir(temporary)
                        break
                    except FileNotFoundError:
                        continue
            _find_dataset_dir(temporary)
            os.replace(temporary, extracted)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    else:
        with zipfile.ZipFile(archive_path) as archive:
            outer_members = [info.filename for info in archive.infolist()]

    manifest: dict[str, object] = {
        "schema_version": 1,
        "url": UCI_HAR_URL,
        "source_url": UCI_HAR_URL,
        "download_url": download_url,
        "archive_name": archive_path.name,
        "archive_sha256": _sha256_file(archive_path),
        "archive_size": archive_path.stat().st_size,
        "members": sorted(outer_members),
        "nested_members": {name: sorted(members) for name, members in sorted(nested_members.items())},
    }
    _write_json_atomic(root / "source_manifest.json", manifest)
    return manifest


def _load_matrix(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, dtype=np.float32))


def _load_vector(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=np.int64))


def _load_partition(dataset_root: Path, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inertial = dataset_root / partition / "Inertial Signals"
    channels = [_load_matrix(inertial / f"{name}_{partition}.txt") for name in SIGNAL_NAMES]
    shapes = {channel.shape for channel in channels}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent {partition} inertial signal shapes: {sorted(shapes)}")
    signal = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    if signal.shape[1:] != (128, len(SIGNAL_NAMES)):
        raise ValueError(f"{partition} signals must be N x 128 x 9, got {signal.shape}")
    target = _load_vector(dataset_root / partition / f"y_{partition}.txt") - 1
    subjects = _load_vector(dataset_root / partition / f"subject_{partition}.txt")
    if len(target) != len(signal) or len(subjects) != len(signal):
        raise ValueError(f"{partition} signals, labels, and subjects have inconsistent lengths")
    if np.any((target < 0) | (target >= 6)):
        raise ValueError(f"{partition} activity labels must be in 1..6 before zero indexing")
    return signal, target.astype(np.int64), subjects.astype(np.int64)


def _fixture_source_manifest(root: Path) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "url": None,
        "source_url": None,
        "download_url": None,
        "archive_name": None,
        "archive_sha256": None,
        "archive_size": 0,
        "members": [],
        "nested_members": {},
    }
    _write_json_atomic(root / "source_manifest.json", manifest)
    return manifest


def prepare_uci_har(
    root: str | Path,
    data_seed: int = 20260716,
    *,
    download_url: str = UCI_HAR_URL,
) -> dict[str, object]:
    """Prepare normalized NPZ splits using a fixed held-out subject subset."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        dataset_root = _find_dataset_dir(root)
    except FileNotFoundError:
        download_uci_har(root, download_url=download_url)
        dataset_root = _find_dataset_dir(root)
    source_path = root / "source_manifest.json"
    if not source_path.exists():
        _fixture_source_manifest(root)

    official_train, train_target, train_subject = _load_partition(dataset_root, "train")
    official_test, test_target, test_subject = _load_partition(dataset_root, "test")
    unique_train_subjects = np.unique(train_subject)
    validation_count = max(1, int(round(0.2 * len(unique_train_subjects))))
    rng = np.random.default_rng(data_seed)
    validation_subjects = np.sort(
        rng.choice(unique_train_subjects, size=validation_count, replace=False)
    )
    is_validation = np.isin(train_subject, validation_subjects)
    train_indices = np.flatnonzero(~is_validation)
    validation_indices = np.flatnonzero(is_validation)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("subject split must leave non-empty training and validation sets")

    normalization_mean = official_train[train_indices].mean(axis=(0, 1), dtype=np.float64)
    normalization_std = official_train[train_indices].std(axis=(0, 1), dtype=np.float64)
    normalization_std = np.maximum(normalization_std, 1e-6)

    def normalize(signal: np.ndarray) -> np.ndarray:
        return ((signal - normalization_mean) / normalization_std).astype(np.float32)

    split_sources = {
        "train": (normalize(official_train[train_indices]), train_target[train_indices], train_subject[train_indices], train_indices, "train"),
        "val": (normalize(official_train[validation_indices]), train_target[validation_indices], train_subject[validation_indices], validation_indices, "train"),
        "test": (normalize(official_test), test_target, test_subject, np.arange(len(official_test)), "test"),
    }
    files: dict[str, dict[str, object]] = {}
    subject_sets: dict[str, list[int]] = {}
    for split, (signal, target, subjects, source_indices, source_partition) in split_sources.items():
        arrays = {
            "sample_id": np.asarray(
                [f"official-{source_partition}-{int(index):06d}" for index in source_indices]
            ),
            "signal": signal,
            "source_index": np.asarray(source_indices, dtype=np.int32),
            "subject": np.asarray(subjects, dtype=np.int16),
            "target": np.asarray(target, dtype=np.int64),
        }
        path = root / f"{split}.npz"
        _write_npz_atomic(path, arrays)
        files[split] = {
            "name": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "shape": list(signal.shape),
        }
        subject_sets[split] = sorted(np.unique(subjects).astype(int).tolist())

    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_seed": data_seed,
        "source_manifest_sha256": _sha256_file(source_path),
        "signal_names": list(SIGNAL_NAMES),
        "official_shapes": {
            "train": list(official_train.shape),
            "test": list(official_test.shape),
        },
        "validation_subjects": validation_subjects.astype(int).tolist(),
        "subjects": subject_sets,
        "normalization_mean": normalization_mean.tolist(),
        "normalization_std": normalization_std.tolist(),
        "files": files,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json_atomic(root / "manifest.json", manifest)
    return manifest


class UCIHARDataset:
    """Prepared UCI HAR split with deterministic time-order controls."""

    def __init__(self, root: str | Path, split: str, transform: str = "none") -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        if transform not in {
            "none",
            "reverse",
            "shuffle",
            "prefix50",
            "noise_025",
        }:
            raise ValueError(
                "transform must be none, reverse, shuffle, prefix50, or noise_025"
            )
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.data_seed = int(self.manifest["data_seed"])
        with np.load(self.root / f"{split}.npz", allow_pickle=False) as data:
            self._arrays = {name: data[name] for name in data.files}

    def __len__(self) -> int:
        return int(len(self._arrays["target"]))

    def _transform_signal(self, signal: np.ndarray, sample_id: str) -> np.ndarray:
        if self.transform == "reverse":
            return signal[::-1].copy()
        if self.transform == "shuffle":
            seed_bytes = hashlib.sha256(f"{self.data_seed}:{sample_id}".encode("utf-8")).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed_bytes, "little", signed=False))
            return signal[rng.permutation(signal.shape[0])].copy()
        if self.transform == "prefix50":
            return signal[:64].copy()
        if self.transform == "noise_025":
            seed_bytes = hashlib.sha256(
                f"{self.data_seed}:{sample_id}:noise_025".encode("utf-8")
            ).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed_bytes, "little", signed=False))
            noise = rng.normal(0.0, 0.25, size=signal.shape)
            return (signal + noise).astype(np.float32)
        return signal.copy()

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = str(self._arrays["sample_id"][index])
        signal = self._transform_signal(self._arrays["signal"][index], sample_id)
        target = np.int64(self._arrays["target"][index])
        coordinate_targets = np.zeros(
            (signal.shape[0], 3, signal.shape[1]), dtype=np.float32
        )
        coordinate_targets[:, 0] = signal
        coordinate_targets[1:, 1] = signal[1:] - signal[:-1]
        coordinate_targets[2:, 2] = signal[2:] - 2 * signal[1:-1] + signal[:-2]
        coordinate_mask = np.ones((signal.shape[0], 3, 1), dtype=np.bool_)
        coordinate_mask[0, 1:] = False
        if signal.shape[0] > 1:
            coordinate_mask[1, 2] = False
        time = np.linspace(0.0, 1.0, signal.shape[0], dtype=np.float32)[:, None]
        features = np.concatenate([signal, time], axis=-1).astype(np.float32, copy=False)
        return {
            "features": features,
            "signal": signal,
            "coordinate_targets": coordinate_targets,
            "coordinate_mask": coordinate_mask,
            "target": target,
            "sample_id": sample_id,
            "formula_family": "",
            "base_target": target,
            "subject_id": np.int64(self._arrays["subject"][index]),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--download-url", default=UCI_HAR_URL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = prepare_uci_har(
        args.root,
        data_seed=args.data_seed,
        download_url=args.download_url,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
