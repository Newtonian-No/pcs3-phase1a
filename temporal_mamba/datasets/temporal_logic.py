"""Verified synthetic benchmark for query-conditioned temporal logic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


FORMULA_FAMILIES = (
    "EVENTUALLY",
    "BEFORE",
    "UNTIL",
    "BOUNDED_RESPONSE",
    "COUNT_WITHIN",
    "GAP",
)
DEFAULT_SIZES = {
    "train": 12_000,
    "val": 2_400,
    "test": 2_400,
    "long_test": 2_400,
}
_SPLITS = tuple(DEFAULT_SIZES)


@dataclass(frozen=True)
class TemporalQuery:
    family: str
    event_a: int
    event_b: int = -1
    p0: int = 0
    p1: int = 0
    p2: int = 0


def _validate_query(signal: np.ndarray, query: TemporalQuery) -> None:
    if signal.ndim != 2:
        raise ValueError(f"signal must be T x D, got shape {signal.shape}")
    if query.family not in FORMULA_FAMILIES:
        raise ValueError(f"unknown formula family: {query.family}")
    if not 0 <= query.event_a < signal.shape[1]:
        raise ValueError(f"event_a out of range: {query.event_a}")
    if query.event_b >= signal.shape[1]:
        raise ValueError(f"event_b out of range: {query.event_b}")


def evaluate_query(signal: np.ndarray, query: TemporalQuery) -> bool:
    """Evaluate one temporal formula with explicit causal index semantics."""

    signal = np.asarray(signal)
    _validate_query(signal, query)
    length = signal.shape[0]
    a = signal[:, query.event_a] > 0.5
    b = None if query.event_b < 0 else signal[:, query.event_b] > 0.5

    if query.family == "EVENTUALLY":
        lo = max(0, min(int(query.p0), length - 1))
        hi = max(lo, min(int(query.p1), length - 1))
        return bool(a[lo : hi + 1].any())

    if query.family == "BEFORE":
        assert b is not None
        a_indices = np.flatnonzero(a)
        b_indices = np.flatnonzero(b)
        return bool(len(a_indices) and len(b_indices) and a_indices[0] < b_indices[0])

    if query.family == "UNTIL":
        assert b is not None
        b_indices = np.flatnonzero(b)
        if not len(b_indices):
            return False
        first_b = int(b_indices[0])
        return bool(a[:first_b].all())

    if query.family == "BOUNDED_RESPONSE":
        assert b is not None
        horizon = int(query.p0)
        if horizon < 1:
            return False
        triggers = np.flatnonzero(a)
        if not len(triggers):
            return False
        for trigger in triggers:
            stop = min(int(trigger) + horizon + 1, length)
            if not b[int(trigger) + 1 : stop].any():
                return False
        return True

    if query.family == "COUNT_WITHIN":
        lo = max(0, min(int(query.p0), length - 1))
        hi = max(lo, min(int(query.p1), length - 1))
        threshold = max(1, int(query.p2))
        return bool(int(a[lo : hi + 1].sum()) >= threshold)

    if query.family == "GAP":
        assert b is not None
        a_indices = np.flatnonzero(a)
        if not len(a_indices):
            return False
        first_a = int(a_indices[0])
        later_b = np.flatnonzero(b[first_a + 1 :])
        if not len(later_b):
            return False
        gap = int(later_b[0]) + 1
        return bool(int(query.p0) <= gap <= int(query.p1))

    raise AssertionError("unreachable")


def encode_query(query: TemporalQuery, event_dim: int, seq_len: int) -> np.ndarray:
    if event_dim <= 0 or seq_len <= 0:
        raise ValueError("event_dim and seq_len must be positive")
    if query.family not in FORMULA_FAMILIES:
        raise ValueError(f"unknown formula family: {query.family}")
    if not 0 <= query.event_a < event_dim:
        raise ValueError(f"event_a out of range: {query.event_a}")
    if query.event_b >= event_dim:
        raise ValueError(f"event_b out of range: {query.event_b}")
    family = np.zeros(len(FORMULA_FAMILIES), dtype=np.float32)
    family[FORMULA_FAMILIES.index(query.family)] = 1.0
    a = np.eye(event_dim, dtype=np.float32)[query.event_a]
    b = (
        np.zeros(event_dim, dtype=np.float32)
        if query.event_b < 0
        else np.eye(event_dim, dtype=np.float32)[query.event_b]
    )
    bounds = np.asarray([query.p0, query.p1, query.p2], dtype=np.float32)
    bounds /= max(seq_len - 1, 1)
    return np.concatenate([family, a, b, bounds]).astype(np.float32, copy=False)


def _other_event(rng: np.random.Generator, event_a: int, event_dim: int) -> int:
    candidate = int(rng.integers(event_dim - 1))
    return candidate + int(candidate >= event_a)


def _construct_sample(
    rng: np.random.Generator,
    family: str,
    label: bool,
    seq_len: int,
    event_dim: int,
) -> tuple[np.ndarray, TemporalQuery]:
    if event_dim < 2:
        raise ValueError("event_dim must be at least 2")
    if seq_len < 16:
        raise ValueError("seq_len must be at least 16")
    event_a = int(rng.integers(event_dim))
    event_b = _other_event(rng, event_a, event_dim)
    signal = np.zeros((seq_len, event_dim), dtype=np.float32)

    if family == "EVENTUALLY":
        lo, hi = seq_len // 4, 3 * seq_len // 4
        query = TemporalQuery(family, event_a, p0=lo, p1=hi)
        if label:
            signal[int(rng.integers(lo, hi + 1)), event_a] = 1.0
        else:
            outside = int(rng.integers(0, lo)) if rng.random() < 0.5 else int(rng.integers(hi + 1, seq_len))
            signal[outside, event_a] = 1.0

    elif family == "BEFORE":
        early = int(rng.integers(1, max(2, seq_len // 3)))
        late = int(rng.integers(2 * seq_len // 3, seq_len - 1))
        query = TemporalQuery(family, event_a, event_b)
        first, second = (event_a, event_b) if label else (event_b, event_a)
        signal[early, first] = 1.0
        signal[late, second] = 1.0

    elif family == "UNTIL":
        first_b = int(rng.integers(seq_len // 3, 2 * seq_len // 3))
        query = TemporalQuery(family, event_a, event_b)
        signal[:first_b, event_a] = 1.0
        signal[first_b, event_b] = 1.0
        if not label:
            signal[int(rng.integers(0, first_b)), event_a] = 0.0

    elif family == "BOUNDED_RESPONSE":
        horizon = max(2, min(8, seq_len // 8))
        query = TemporalQuery(family, event_a, event_b, p0=horizon)
        if label:
            triggers = (seq_len // 4, seq_len // 2)
            for trigger in triggers:
                signal[trigger, event_a] = 1.0
                delay = int(rng.integers(1, horizon + 1))
                signal[trigger + delay, event_b] = 1.0
        else:
            trigger = seq_len // 3
            signal[trigger, event_a] = 1.0
            signal[trigger + horizon + 2, event_b] = 1.0

    elif family == "COUNT_WITHIN":
        lo, hi, threshold = seq_len // 4, 3 * seq_len // 4, 3
        query = TemporalQuery(family, event_a, p0=lo, p1=hi, p2=threshold)
        count = threshold if label else threshold - 1
        points = rng.choice(np.arange(lo, hi + 1), size=count, replace=False)
        signal[points, event_a] = 1.0
        if not label:
            signal[int(rng.integers(0, lo)), event_a] = 1.0

    elif family == "GAP":
        low = max(2, seq_len // 16)
        high = max(low, seq_len // 8)
        query = TemporalQuery(family, event_a, event_b, p0=low, p1=high)
        start = seq_len // 4
        gap = int(rng.integers(low, high + 1)) if label else high + 2
        signal[start, event_a] = 1.0
        signal[start + gap, event_b] = 1.0

    else:
        raise ValueError(f"unknown formula family: {family}")

    relevant = {event_a}
    if query.event_b >= 0:
        relevant.add(query.event_b)
    for channel in range(event_dim):
        if channel not in relevant:
            signal[:, channel] = (rng.random(seq_len) < 0.035).astype(np.float32)
    return signal, query


def _sample_fingerprint(signal: np.ndarray, query: TemporalQuery, target: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(signal).tobytes())
    digest.update(
        np.asarray(
            [FORMULA_FAMILIES.index(query.family), query.event_a, query.event_b, query.p0, query.p1, query.p2, target],
            dtype="<i8",
        ).tobytes()
    )
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-reproducible NPZ by fixing member order and ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_temporal_logic_manifest(
    root: str | Path,
    sizes: Mapping[str, int] | None = None,
    data_seed: int = 20260716,
    *,
    event_dim: int = 8,
    seq_len: int = 128,
    long_seq_len: int = 256,
) -> dict[str, object]:
    """Construct, verify, and persist all immutable benchmark splits."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sizes = dict(DEFAULT_SIZES if sizes is None else sizes)
    if set(sizes) != set(_SPLITS):
        raise ValueError(f"sizes must contain exactly {_SPLITS}")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes.values()):
        raise ValueError("every split size must be a positive integer")
    if data_seed <= 0:
        raise ValueError("data_seed must be positive")

    child_sequences = np.random.SeedSequence(data_seed).spawn(len(_SPLITS))
    all_fingerprints: dict[str, set[str]] = {}
    files: dict[str, dict[str, object]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for split, child_sequence in zip(_SPLITS, child_sequences):
        length = long_seq_len if split == "long_test" else seq_len
        rng = np.random.default_rng(child_sequence)
        records: list[tuple[np.ndarray, TemporalQuery, int, str]] = []
        fingerprints: set[str] = set()
        for index in range(sizes[split]):
            family_index = (index % (2 * len(FORMULA_FAMILIES))) // 2
            target = index % 2
            family = FORMULA_FAMILIES[family_index]
            for _ in range(1_000):
                signal, query = _construct_sample(rng, family, bool(target), length, event_dim)
                if evaluate_query(signal, query) is not bool(target):
                    continue
                fingerprint = _sample_fingerprint(signal, query, target)
                if fingerprint in fingerprints or any(fingerprint in prior for prior in all_fingerprints.values()):
                    continue
                records.append((signal, query, target, fingerprint))
                fingerprints.add(fingerprint)
                break
            else:
                raise RuntimeError(f"failed to construct unique verified sample for {split}:{index}")

        order = rng.permutation(len(records))
        records = [records[int(i)] for i in order]
        arrays = {
            "event_a": np.asarray([item[1].event_a for item in records], dtype=np.int16),
            "event_b": np.asarray([item[1].event_b for item in records], dtype=np.int16),
            "family": np.asarray([FORMULA_FAMILIES.index(item[1].family) for item in records], dtype=np.int8),
            "p0": np.asarray([item[1].p0 for item in records], dtype=np.int16),
            "p1": np.asarray([item[1].p1 for item in records], dtype=np.int16),
            "p2": np.asarray([item[1].p2 for item in records], dtype=np.int16),
            "sample_id": np.asarray([f"{split}-{i:06d}" for i in range(len(records))]),
            "signal": np.stack([item[0] for item in records]).astype(np.float32, copy=False),
            "target": np.asarray([item[2] for item in records], dtype=np.int8),
        }
        path = root / f"{split}.npz"
        _write_npz_atomic(path, arrays)
        files[split] = {
            "name": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "shape": [sizes[split], length, event_dim],
            "child_spawn_key": list(child_sequence.spawn_key),
        }
        split_counts: dict[str, dict[str, int]] = {}
        for family_index, family in enumerate(FORMULA_FAMILIES):
            labels = arrays["target"][arrays["family"] == family_index]
            split_counts[family] = {
                "negative": int((labels == 0).sum()),
                "positive": int((labels == 1).sum()),
            }
        counts[split] = split_counts
        all_fingerprints[split] = fingerprints

    cross_split_duplicates = 0
    for i, split_a in enumerate(_SPLITS):
        for split_b in _SPLITS[i + 1 :]:
            cross_split_duplicates += len(all_fingerprints[split_a] & all_fingerprints[split_b])

    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_seed": data_seed,
        "event_dim": event_dim,
        "seq_len": seq_len,
        "long_seq_len": long_seq_len,
        "formula_families": list(FORMULA_FAMILIES),
        "sizes": {split: sizes[split] for split in _SPLITS},
        "files": files,
        "counts": counts,
        "cross_split_duplicates": cross_split_duplicates,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json_atomic(root / "manifest.json", manifest)
    return manifest


class TemporalLogicDataset:
    """Deterministic original/reverse/shuffle views over a generated split."""

    def __init__(self, root: str | Path, split: str, transform: str = "none") -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")
        if transform not in {"none", "reverse", "shuffle"}:
            raise ValueError("transform must be none, reverse, or shuffle")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.data_seed = int(self.manifest["data_seed"])
        with np.load(self.root / f"{split}.npz", allow_pickle=False) as data:
            self._arrays = {name: data[name] for name in data.files}

    def __len__(self) -> int:
        return int(len(self._arrays["target"]))

    def query_at(self, index: int) -> TemporalQuery:
        return TemporalQuery(
            family=FORMULA_FAMILIES[int(self._arrays["family"][index])],
            event_a=int(self._arrays["event_a"][index]),
            event_b=int(self._arrays["event_b"][index]),
            p0=int(self._arrays["p0"][index]),
            p1=int(self._arrays["p1"][index]),
            p2=int(self._arrays["p2"][index]),
        )

    def _transform_signal(self, signal: np.ndarray, sample_id: str) -> np.ndarray:
        if self.transform == "reverse":
            return signal[::-1].copy()
        if self.transform == "shuffle":
            seed_bytes = hashlib.sha256(f"{self.data_seed}:{sample_id}".encode("utf-8")).digest()[:8]
            rng = np.random.default_rng(int.from_bytes(seed_bytes, "little", signed=False))
            return signal[rng.permutation(signal.shape[0])].copy()
        return signal.copy()

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = str(self._arrays["sample_id"][index])
        signal = self._transform_signal(self._arrays["signal"][index], sample_id)
        query = self.query_at(index)
        base_target = np.float32(self._arrays["target"][index])
        target = np.float32(evaluate_query(signal, query))
        length, event_dim = signal.shape
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
        query_features = np.broadcast_to(
            encode_query(query, event_dim=event_dim, seq_len=length),
            (length, len(FORMULA_FAMILIES) + 2 * event_dim + 3),
        )
        features = np.concatenate([signal, time, query_features], axis=-1).astype(np.float32, copy=False)
        return {
            "features": features,
            "signal": signal,
            "target": target,
            "sample_id": sample_id,
            "formula_family": query.family,
            "base_target": base_target,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260716)
    parser.add_argument("--train-size", type=int, default=DEFAULT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument("--long-test-size", type=int, default=DEFAULT_SIZES["long_test"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_temporal_logic_manifest(
        args.root,
        {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
            "long_test": args.long_test_size,
        },
        data_seed=args.data_seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
