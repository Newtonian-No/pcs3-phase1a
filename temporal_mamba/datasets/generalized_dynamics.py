"""Deterministic analytic second-order dynamics benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from temporal_mamba.config import load_experiment_config

from .temporal_logic import _sha256_file, _write_json_atomic, _write_npz_atomic


DYNAMICS_SPLITS = (
    "train",
    "val",
    "test",
    "length_256",
    "length_512",
    "parameter_ood",
    "noise_ood",
)
FORMULA_FAMILIES = ("damped", "forced", "switching")
GENERATOR_VERSION = "generalized-dynamics-v1"
NOISE_STD = 0.25

_ID_RANGES: dict[str, list[float]] = {
    "amplitude": [0.75, 1.25],
    "phase": [-float(np.pi), float(np.pi)],
    "damping": [0.05, 0.20],
    "omega": [1.0, 3.0],
    "drive_omega": [3.5, 5.0],
    "switch_omega_before": [1.0, 2.0],
    "switch_rate": [1.25, 1.75],
    "switch_fraction": [0.35, 0.65],
}
_PARAMETER_OOD_RANGES: dict[str, list[float]] = {
    "amplitude": [0.75, 1.25],
    "phase": [-float(np.pi), float(np.pi)],
    "damping": [0.30, 0.50],
    "omega": [4.0, 6.0],
    "drive_omega": [6.5, 8.0],
    "switch_omega_before": [3.5, 5.0],
    "switch_rate": [2.5, 3.5],
    "switch_fraction": [0.10, 0.25],
}


def damped(
    t: np.ndarray,
    amp: float,
    phase: float,
    damping: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = omega * t + phase
    decay = np.exp(-damping * t)
    x = amp * decay * np.cos(angle)
    dx = amp * decay * (-damping * np.cos(angle) - omega * np.sin(angle))
    ddx = -2 * damping * dx - (damping**2 + omega**2) * x
    return x, dx, ddx


def forced(
    t: np.ndarray,
    amp: float,
    phase: float,
    omega: float,
    drive_omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = amp * np.cos(omega * t + phase) + 0.5 * amp * np.cos(drive_omega * t - phase)
    dx = (
        -amp * omega * np.sin(omega * t + phase)
        - 0.5 * amp * drive_omega * np.sin(drive_omega * t - phase)
    )
    ddx = (
        -amp * omega**2 * np.cos(omega * t + phase)
        - 0.5 * amp * drive_omega**2 * np.cos(drive_omega * t - phase)
    )
    return x, dx, ddx


def switching(
    t: np.ndarray,
    amp: float,
    phase: float,
    omega_before: float,
    omega_after: float,
    switch_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    switch_t = t[switch_index]
    angle_before = omega_before * t + phase
    switch_angle = omega_before * switch_t + phase
    angle_after = switch_angle + omega_after * (t - switch_t)
    before = np.arange(len(t)) < switch_index
    angle = np.where(before, angle_before, angle_after)
    omega = np.where(before, omega_before, omega_after)
    x = amp * np.cos(angle)
    dx = -amp * omega * np.sin(angle)
    ddx = -amp * omega**2 * np.cos(angle)
    return x, dx, ddx


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _split_length(split: str, seq_len: int) -> int:
    if split == "length_256":
        return 256
    if split == "length_512":
        return 512
    return seq_len


def _generate_raw_split(
    split: str,
    size: int,
    signal_dim: int,
    seq_len: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    length = _split_length(split, seq_len)
    t = np.linspace(0.0, 10.0, length, dtype=np.float64)
    ranges = _PARAMETER_OOD_RANGES if split == "parameter_ood" else _ID_RANGES
    family = np.repeat(np.arange(len(FORMULA_FAMILIES), dtype=np.int8), size // 3)
    family = family[rng.permutation(size)]
    coordinates = np.empty((size, length, 3, signal_dim), dtype=np.float64)
    switch_index = np.full(size, -1, dtype=np.int32)
    switch_position = np.full(size, np.nan, dtype=np.float64)
    switch_rate = np.full((size, signal_dim), np.nan, dtype=np.float64)

    for sample in range(size):
        family_index = int(family[sample])
        sample_switch_index = -1
        if family_index == 2:
            fraction = _uniform(rng, ranges["switch_fraction"])
            sample_switch_index = int(np.clip(round(fraction * (length - 1)), 1, length - 1))
            switch_index[sample] = sample_switch_index
            switch_position[sample] = float(t[sample_switch_index])
        for channel in range(signal_dim):
            amp = _uniform(rng, ranges["amplitude"])
            phase = _uniform(rng, ranges["phase"])
            if family_index == 0:
                values = damped(
                    t,
                    amp,
                    phase,
                    _uniform(rng, ranges["damping"]),
                    _uniform(rng, ranges["omega"]),
                )
            elif family_index == 1:
                values = forced(
                    t,
                    amp,
                    phase,
                    _uniform(rng, ranges["omega"]),
                    _uniform(rng, ranges["drive_omega"]),
                )
            else:
                omega_before = _uniform(rng, ranges["switch_omega_before"])
                rate = _uniform(rng, ranges["switch_rate"])
                switch_rate[sample, channel] = rate
                values = switching(
                    t,
                    amp,
                    phase,
                    omega_before,
                    omega_before * rate,
                    sample_switch_index,
                )
            coordinates[sample, :, :, channel] = np.stack(values, axis=-1)

    return {
        "coordinate_targets": coordinates,
        "family": family,
        "parameter_regime": np.full(size, split == "parameter_ood", dtype=np.int8),
        "switch_index": switch_index,
        "switch_position": switch_position,
        "switch_rate": switch_rate,
    }


def _normalize_coordinates(
    coordinate_targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    normalized = coordinate_targets.copy()
    normalized[:, :, 0, :] = (normalized[:, :, 0, :] - mean) / std
    normalized[:, :, 1:, :] /= std[None, None, None, :]
    return normalized


def _content_fingerprint(coordinates: np.ndarray, target: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coordinates[:, 0, :]).tobytes())
    digest.update(np.asarray([target], dtype="<i8").tobytes())
    return digest.hexdigest()


def _validate_builder_inputs(
    sizes: Mapping[str, int],
    data_seed: int,
    signal_dim: int,
    seq_len: int,
) -> dict[str, int]:
    sizes = dict(sizes)
    if set(sizes) != set(DYNAMICS_SPLITS):
        raise ValueError(f"sizes must contain exactly {DYNAMICS_SPLITS}")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes.values()):
        raise ValueError("every split size must be a positive integer")
    if any(size % 3 for size in sizes.values()):
        raise ValueError("every split size must be a multiple of three")
    if isinstance(data_seed, bool) or not isinstance(data_seed, int) or data_seed <= 0:
        raise ValueError("data_seed must be a positive integer")
    if isinstance(signal_dim, bool) or not isinstance(signal_dim, int) or signal_dim <= 0:
        raise ValueError("signal_dim must be a positive integer")
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 16:
        raise ValueError("seq_len must be an integer of at least 16")
    return sizes


def build_generalized_dynamics_manifest(
    root: str | Path,
    data_seed: int,
    sizes: Mapping[str, int],
    signal_dim: int = 6,
    seq_len: int = 128,
) -> dict[str, object]:
    """Generate immutable balanced analytic dynamics splits and their strict manifest."""

    sizes = _validate_builder_inputs(sizes, data_seed, signal_dim, seq_len)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    child_sequences = np.random.SeedSequence(data_seed).spawn(len(DYNAMICS_SPLITS))

    train_raw = _generate_raw_split(
        "train",
        sizes["train"],
        signal_dim,
        seq_len,
        np.random.default_rng(child_sequences[0]),
    )
    train_signal = train_raw["coordinate_targets"][:, :, 0, :]
    mean = train_signal.mean(axis=(0, 1), dtype=np.float64)
    std = train_signal.std(axis=(0, 1), dtype=np.float64)
    if np.any(~np.isfinite(std)) or np.any(std <= np.finfo(np.float64).eps):
        raise RuntimeError("training signal has a degenerate normalization channel")

    files: dict[str, dict[str, object]] = {}
    label_counts: dict[str, dict[str, int]] = {}
    fingerprints_by_split: dict[str, set[str]] = {}
    sample_ids_by_split: dict[str, set[str]] = {}
    for split_index, (split, child_sequence) in enumerate(zip(DYNAMICS_SPLITS, child_sequences)):
        raw = train_raw if split == "train" else _generate_raw_split(
            split,
            sizes[split],
            signal_dim,
            seq_len,
            np.random.default_rng(child_sequence),
        )
        coordinates = _normalize_coordinates(raw["coordinate_targets"], mean, std).astype(np.float32)
        clean_signal = coordinates[:, :, 0, :].copy()
        signal = clean_signal.copy()
        if split == "noise_ood":
            noise_sequence = child_sequence.spawn(1)[0]
            noise_rng = np.random.default_rng(noise_sequence)
            signal += noise_rng.normal(0.0, NOISE_STD, size=signal.shape).astype(np.float32)

        sample_ids = np.asarray([f"{split}-{index:06d}" for index in range(sizes[split])])
        fingerprints = np.asarray(
            [
                _content_fingerprint(coordinates[index], int(raw["family"][index]))
                for index in range(sizes[split])
            ]
        )
        if len(set(fingerprints.tolist())) != sizes[split]:
            raise RuntimeError(f"duplicate content fingerprint within split: {split}")
        arrays = {
            "clean_signal": clean_signal,
            "coordinate_targets": coordinates,
            "family": raw["family"].astype(np.int8, copy=False),
            "fingerprint": fingerprints,
            "parameter_regime": raw["parameter_regime"].astype(np.int8, copy=False),
            "sample_id": sample_ids,
            "signal": signal,
            "switch_index": raw["switch_index"].astype(np.int32, copy=False),
            "switch_position": raw["switch_position"].astype(np.float32),
            "switch_rate": raw["switch_rate"].astype(np.float32),
            "target": raw["family"].astype(np.int8, copy=False),
        }
        path = root / f"{split}.npz"
        _write_npz_atomic(path, arrays)
        length = _split_length(split, seq_len)
        files[split] = {
            "name": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "shape": [sizes[split], length, signal_dim],
            "child_spawn_key": list(child_sequence.spawn_key),
        }
        label_counts[split] = {
            str(label): int(np.count_nonzero(raw["family"] == label))
            for label in range(len(FORMULA_FAMILIES))
        }
        fingerprints_by_split[split] = set(fingerprints.tolist())
        sample_ids_by_split[split] = set(sample_ids.tolist())
        if split_index == 0:
            del train_raw

    duplicate_count = 0
    sample_id_duplicate_count = 0
    for index, split_a in enumerate(DYNAMICS_SPLITS):
        for split_b in DYNAMICS_SPLITS[index + 1 :]:
            duplicate_count += len(fingerprints_by_split[split_a] & fingerprints_by_split[split_b])
            sample_id_duplicate_count += len(sample_ids_by_split[split_a] & sample_ids_by_split[split_b])
    if duplicate_count:
        raise RuntimeError("generated content fingerprints overlap across splits")
    if sample_id_duplicate_count:
        raise RuntimeError("generated sample IDs overlap across splits")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "data_seed": data_seed,
        "signal_dim": signal_dim,
        "seq_len": seq_len,
        "formula_families": list(FORMULA_FAMILIES),
        "splits": list(DYNAMICS_SPLITS),
        "sizes": {split: sizes[split] for split in DYNAMICS_SPLITS},
        "shapes": {
            "signal": [None, signal_dim],
            "coordinate_targets": [None, 3, signal_dim],
            "coordinate_mask": [None, 3, 1],
            "features": [None, signal_dim + 1],
        },
        "ranges": {
            "id": _ID_RANGES,
            "parameter_ood": _PARAMETER_OOD_RANGES,
            "noise_ood": {"observation_noise_std": NOISE_STD},
        },
        "normalization": {
            "source_split": "train",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "derivative_scale": std.tolist(),
        },
        "files": files,
        "label_counts": label_counts,
        "cross_split_duplicates": duplicate_count,
        "cross_split_sample_id_duplicates": sample_id_duplicate_count,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json_atomic(root / "manifest.json", manifest)
    return manifest


class GeneralizedDynamicsDataset:
    """Read-only view over one immutable analytic dynamics split."""

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in DYNAMICS_SPLITS:
            raise ValueError(f"split must be one of {DYNAMICS_SPLITS}")
        self.root = Path(root)
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        claimed_manifest_hash = self.manifest.get("manifest_sha256")
        canonical_manifest = dict(self.manifest)
        canonical_manifest.pop("manifest_sha256", None)
        actual_manifest_hash = hashlib.sha256(
            json.dumps(canonical_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if claimed_manifest_hash != actual_manifest_hash:
            raise ValueError("manifest_sha256 does not match canonical manifest contents")
        file_entry = self.manifest["files"][split]
        path = self.root / file_entry["name"]
        if _sha256_file(path) != file_entry["sha256"]:
            raise ValueError(f"split file hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as data:
            self._arrays = {name: data[name] for name in data.files}

    def __len__(self) -> int:
        return int(len(self._arrays["target"]))

    def __getitem__(self, index: int) -> dict[str, object]:
        signal = self._arrays["signal"][index].copy()
        coordinates = self._arrays["coordinate_targets"][index].copy()
        length = signal.shape[0]
        time = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
        features = np.concatenate([signal, time], axis=-1).astype(np.float32, copy=False)
        target = int(self._arrays["target"][index])
        return {
            "features": features,
            "signal": signal,
            "coordinate_targets": coordinates,
            "coordinate_mask": np.ones((length, 3, 1), dtype=np.float32),
            "target": target,
            "base_target": target,
            "sample_id": str(self._arrays["sample_id"][index]),
            "formula_family": FORMULA_FAMILIES[target],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config, variant="gc_k3", seed=42)
    if config.dataset != "generalized_dynamics":
        raise ValueError("config dataset must be generalized_dynamics")
    sizes = {
        "train": config.data.train_size,
        "val": config.data.val_size,
        "test": config.data.test_size,
        "length_256": config.data.long_test_size,
        "length_512": config.data.long_test_size,
        "parameter_ood": config.data.long_test_size,
        "noise_ood": config.data.long_test_size,
    }
    manifest = build_generalized_dynamics_manifest(
        args.root,
        config.data_seed if args.data_seed is None else args.data_seed,
        sizes,
        signal_dim=config.signal_dim,
        seq_len=config.seq_len,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
