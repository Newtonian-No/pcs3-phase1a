"""Matched temporal-logic benchmark with frozen diagnostic labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .temporal_logic import (
    FORMULA_FAMILIES,
    TemporalQuery,
    _sample_fingerprint,
    _sha256_file,
    _write_json_atomic,
    _write_npz_atomic,
    encode_query,
    evaluate_query,
)


V2_SPLITS = ("train", "val", "test", "long_test", "channel_ood")
DEFAULT_V2_SIZES = {
    "train": 12_000,
    "val": 2_400,
    "test": 2_400,
    "long_test": 2_400,
    "channel_ood": 2_400,
}


def _outside_window(rng: np.random.Generator, lo: int, hi: int, length: int) -> int:
    candidates = np.concatenate((np.arange(0, lo), np.arange(hi + 1, length)))
    return int(rng.choice(candidates))


def _matched_pair(
    rng: np.random.Generator,
    family: str,
    seq_len: int,
    event_dim: int,
    *,
    ood: bool,
    pair_index: int,
) -> tuple[tuple[np.ndarray, TemporalQuery], tuple[np.ndarray, TemporalQuery]]:
    """Return a negative/positive pair with matched relevant global counts."""

    if event_dim < 8:
        raise ValueError("event_dim must be at least 8 for the channel-OOD split")
    if seq_len < 16:
        raise ValueError("seq_len must be at least 16")

    allowed_a = np.arange(6, event_dim) if ood else np.arange(6)
    event_a = int(allowed_a[pair_index % len(allowed_a)])
    other = np.asarray([channel for channel in range(event_dim) if channel != event_a])
    event_b = int(rng.choice(other))
    negative = np.zeros((seq_len, event_dim), dtype=np.float32)
    positive = np.zeros_like(negative)

    if family == "EVENTUALLY":
        lo, hi = seq_len // 4, 3 * seq_len // 4
        query = TemporalQuery(family, event_a, p0=lo, p1=hi)
        positive[int(rng.integers(lo, hi + 1)), event_a] = 1.0
        negative[_outside_window(rng, lo, hi, seq_len), event_a] = 1.0

    elif family == "BEFORE":
        early = int(rng.integers(1, max(2, seq_len // 3)))
        late = int(rng.integers(2 * seq_len // 3, seq_len - 1))
        query = TemporalQuery(family, event_a, event_b)
        positive[early, event_a] = positive[late, event_b] = 1.0
        negative[early, event_b] = negative[late, event_a] = 1.0

    elif family == "UNTIL":
        first_b = int(rng.integers(seq_len // 3, 2 * seq_len // 3))
        query = TemporalQuery(family, event_a, event_b)
        positive[:first_b, event_a] = 1.0
        positive[first_b, event_b] = 1.0
        negative[:] = positive
        negative[int(rng.integers(0, first_b)), event_a] = 0.0
        negative[int(rng.integers(first_b + 1, seq_len)), event_a] = 1.0

    elif family == "BOUNDED_RESPONSE":
        horizon = max(2, min(8, seq_len // 8))
        query = TemporalQuery(family, event_a, event_b, p0=horizon)
        triggers = (seq_len // 4, 3 * seq_len // 4)
        delays = [int(rng.integers(1, horizon + 1)) for _ in triggers]
        for trigger, delay in zip(triggers, delays):
            positive[trigger, event_a] = negative[trigger, event_a] = 1.0
            positive[trigger + delay, event_b] = negative[trigger + delay, event_b] = 1.0
        moved_from = triggers[-1] + delays[-1]
        negative[moved_from, event_b] = 0.0
        negative[triggers[-1] + horizon + 1, event_b] = 1.0

    elif family == "COUNT_WITHIN":
        lo, hi, threshold = seq_len // 4, 3 * seq_len // 4, 3
        query = TemporalQuery(family, event_a, p0=lo, p1=hi, p2=threshold)
        points = rng.choice(np.arange(lo, hi + 1), size=threshold, replace=False)
        positive[points, event_a] = 1.0
        negative[points[:-1], event_a] = 1.0
        negative[_outside_window(rng, lo, hi, seq_len), event_a] = 1.0

    elif family == "GAP":
        low = max(2, seq_len // 16)
        high = max(low + 1, seq_len // 8)
        query = TemporalQuery(family, event_a, event_b, p0=low, p1=high)
        start = seq_len // 4
        positive_gap = int(rng.integers(low, high + 1))
        invalid_gaps = [gap for gap in (low - 1, high + 1) if 0 < start + gap < seq_len]
        negative_gap = int(rng.choice(np.asarray(invalid_gaps)))
        positive[start, event_a] = negative[start, event_a] = 1.0
        positive[start + positive_gap, event_b] = 1.0
        negative[start + negative_gap, event_b] = 1.0

    else:
        raise ValueError(f"unknown formula family: {family}")

    relevant = {event_a}
    if query.event_b >= 0:
        relevant.add(query.event_b)
    for channel in range(event_dim):
        if channel not in relevant:
            distractors = (rng.random(seq_len) < 0.035).astype(np.float32)
            negative[:, channel] = distractors
            positive[:, channel] = distractors

    if evaluate_query(negative, query) is not False:
        raise AssertionError(f"v2 {family} negative constructor produced true")
    if evaluate_query(positive, query) is not True:
        raise AssertionError(f"v2 {family} positive constructor produced false")
    return (negative, query), (positive, query)


def build_temporal_logic_v2_manifest(
    root: str | Path,
    sizes: Mapping[str, int] | None = None,
    data_seed: int = 20260717,
    *,
    event_dim: int = 8,
    seq_len: int = 128,
    long_seq_len: int = 256,
) -> dict[str, object]:
    """Construct and persist balanced matched pairs for every immutable split."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sizes = dict(DEFAULT_V2_SIZES if sizes is None else sizes)
    if set(sizes) != set(V2_SPLITS):
        raise ValueError(f"sizes must contain exactly {V2_SPLITS}")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes.values()):
        raise ValueError("every split size must be a positive integer")
    pair_block = 2 * len(FORMULA_FAMILIES)
    if any(size % pair_block for size in sizes.values()):
        raise ValueError(f"every split size must be divisible by {pair_block}")
    if data_seed <= 0:
        raise ValueError("data_seed must be positive")

    child_sequences = np.random.SeedSequence(data_seed).spawn(len(V2_SPLITS))
    all_fingerprints: dict[str, set[str]] = {}
    files: dict[str, dict[str, object]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for split, child_sequence in zip(V2_SPLITS, child_sequences):
        length = long_seq_len if split == "long_test" else seq_len
        rng = np.random.default_rng(child_sequence)
        records: list[tuple[np.ndarray, TemporalQuery, int, str]] = []
        fingerprints: set[str] = set()
        pairs_per_family = sizes[split] // pair_block
        for family in FORMULA_FAMILIES:
            for pair_index in range(pairs_per_family):
                for _ in range(1_000):
                    pair = _matched_pair(
                        rng,
                        family,
                        length,
                        event_dim,
                        ood=split == "channel_ood",
                        pair_index=pair_index,
                    )
                    candidates = []
                    for target, (signal, query) in enumerate(pair):
                        fingerprint = _sample_fingerprint(signal, query, target)
                        candidates.append((signal, query, target, fingerprint))
                    candidate_fingerprints = {item[3] for item in candidates}
                    seen_before = fingerprints | set().union(*all_fingerprints.values()) if all_fingerprints else fingerprints
                    if len(candidate_fingerprints) != 2 or candidate_fingerprints & seen_before:
                        continue
                    records.extend(candidates)
                    fingerprints.update(candidate_fingerprints)
                    break
                else:
                    raise RuntimeError(f"failed to construct unique v2 pair for {split}:{family}:{pair_index}")

        order = rng.permutation(len(records))
        records = [records[int(index)] for index in order]
        arrays = {
            "event_a": np.asarray([item[1].event_a for item in records], dtype=np.int16),
            "event_b": np.asarray([item[1].event_b for item in records], dtype=np.int16),
            "family": np.asarray([FORMULA_FAMILIES.index(item[1].family) for item in records], dtype=np.int8),
            "p0": np.asarray([item[1].p0 for item in records], dtype=np.int16),
            "p1": np.asarray([item[1].p1 for item in records], dtype=np.int16),
            "p2": np.asarray([item[1].p2 for item in records], dtype=np.int16),
            "sample_id": np.asarray([f"{split}-{index:06d}" for index in range(len(records))]),
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
        counts[split] = {}
        for family_index, family in enumerate(FORMULA_FAMILIES):
            labels = arrays["target"][arrays["family"] == family_index]
            counts[split][family] = {
                "negative": int((labels == 0).sum()),
                "positive": int((labels == 1).sum()),
            }
        all_fingerprints[split] = fingerprints

    cross_split_duplicates = 0
    for index, split_a in enumerate(V2_SPLITS):
        for split_b in V2_SPLITS[index + 1 :]:
            cross_split_duplicates += len(all_fingerprints[split_a] & all_fingerprints[split_b])

    manifest: dict[str, object] = {
        "schema_version": 2,
        "data_seed": data_seed,
        "event_dim": event_dim,
        "seq_len": seq_len,
        "long_seq_len": long_seq_len,
        "formula_families": list(FORMULA_FAMILIES),
        "sizes": {split: sizes[split] for split in V2_SPLITS},
        "files": files,
        "counts": counts,
        "cross_split_duplicates": cross_split_duplicates,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json_atomic(root / "manifest.json", manifest)
    return manifest


class TemporalLogicV2Dataset:
    """Original/reverse/shuffle views that retain the original frozen target."""

    def __init__(self, root: str | Path, split: str, transform: str = "none") -> None:
        if split not in V2_SPLITS:
            raise ValueError(f"split must be one of {V2_SPLITS}")
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
        target = np.float32(self._arrays["target"][index])
        length, event_dim = signal.shape
        query_vector = encode_query(query, event_dim=event_dim, seq_len=length)
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
        legacy = np.concatenate(
            [signal, time, np.broadcast_to(query_vector, (length, len(query_vector)))],
            axis=-1,
        ).astype(np.float32, copy=False)
        return {
            "features": legacy,
            "signal": signal.astype(np.float32, copy=False),
            "query": query_vector,
            "target": target,
            "base_target": target,
            "sample_id": sample_id,
            "formula_family": query.family,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260717)
    for split, size in DEFAULT_V2_SIZES.items():
        parser.add_argument(f"--{split.replace('_', '-')}-size", type=int, default=size)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sizes = {split: getattr(args, f"{split}_size") for split in V2_SPLITS}
    manifest = build_temporal_logic_v2_manifest(args.root, sizes, data_seed=args.data_seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

