"""Sequential, resumable launcher for the causal ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .config import TRAINING_SEEDS, VARIANTS, load_experiment_config
from .train import _canonical_hash, _config_payload, _git_commit


DATASETS = ("temporal_logic", "uci_har")


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    variant: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.dataset}-{self.variant}-seed{self.seed}"

    @property
    def config_name(self) -> str:
        return f"{self.dataset}.json"


def expand_matrix(
    *,
    datasets: Sequence[str] = DATASETS,
    variants: Sequence[str] = VARIANTS,
    seeds: Sequence[int] = TRAINING_SEEDS,
) -> list[RunSpec]:
    unknown_datasets = set(datasets) - set(DATASETS)
    unknown_variants = set(variants) - set(VARIANTS)
    unknown_seeds = set(seeds) - set(TRAINING_SEEDS)
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {sorted(unknown_datasets)}")
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    if unknown_seeds:
        raise ValueError(f"unapproved seeds: {sorted(unknown_seeds)}")
    return [
        RunSpec(dataset, variant, int(seed))
        for dataset in datasets
        for variant in variants
        for seed in seeds
    ]


def _append_status(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _expected_metadata(
    spec: RunSpec,
    *,
    config_dir: Path,
    data_root: Path,
    epochs_override: int | None,
) -> dict[str, str] | None:
    config_path = config_dir / spec.config_name
    manifest_path = data_root / spec.dataset / "manifest.json"
    if not config_path.exists() or not manifest_path.exists():
        return None
    config = load_experiment_config(config_path, variant=spec.variant, seed=spec.seed)
    if epochs_override is not None:
        config = replace(config, training=replace(config.training, epochs=epochs_override))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "config_hash": _canonical_hash(_config_payload(config)),
        "dataset_manifest_hash": str(manifest["manifest_sha256"]),
        "git_commit": _git_commit(),
    }


def _is_reusable_final(path: Path, spec: RunSpec, expected: dict[str, str] | None) -> bool:
    if not path.exists() or expected is None:
        return False
    try:
        final = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        final.get("status") == "complete"
        and final.get("run_id") == spec.run_id
        and final.get("dataset") == spec.dataset
        and final.get("variant") == spec.variant
        and final.get("seed") == spec.seed
        and all(final.get(name) == value for name, value in expected.items())
    )


def run_matrix(
    specs: Iterable[RunSpec],
    *,
    artifact_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path = "configs",
    epochs_override: int | None = None,
    device: str | None = None,
    num_workers: int = 0,
    dry_run: bool = False,
) -> list[str]:
    specs = list(specs)
    if dry_run:
        return [spec.run_id for spec in specs]
    artifact_root = Path(artifact_root)
    data_root = Path(data_root)
    config_dir = Path(config_dir)
    status_path = artifact_root / "matrix_status.jsonl"
    completed: list[str] = []
    pycache_root = Path(tempfile.gettempdir()) / f"pcs3-pycache-{_git_commit()[:12]}"

    for spec in specs:
        expected = _expected_metadata(
            spec,
            config_dir=config_dir,
            data_root=data_root,
            epochs_override=epochs_override,
        )
        final_path = artifact_root / spec.run_id / "final.json"
        if _is_reusable_final(final_path, spec, expected):
            _append_status(
                status_path,
                {
                    "run_id": spec.run_id,
                    "event": "skip_verified",
                    "time": datetime.now(timezone.utc).isoformat(),
                },
            )
            completed.append(spec.run_id)
            continue

        command = [
            sys.executable,
            "-m",
            "temporal_mamba.train",
            "--config",
            str(config_dir / spec.config_name),
            "--variant",
            spec.variant,
            "--seed",
            str(spec.seed),
            "--data-root",
            str(data_root / spec.dataset),
            "--artifact-root",
            str(artifact_root),
            "--num-workers",
            str(num_workers),
        ]
        if epochs_override is not None:
            command.extend(["--epochs", str(epochs_override)])
        if device is not None:
            command.extend(["--device", device])
        started = datetime.now(timezone.utc).isoformat()
        _append_status(status_path, {"run_id": spec.run_id, "event": "start", "time": started})
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(pycache_root)
        environment["PYTHONHASHSEED"] = str(spec.seed)
        result = subprocess.run(command, check=False, env=environment)
        ended = datetime.now(timezone.utc).isoformat()
        _append_status(
            status_path,
            {
                "run_id": spec.run_id,
                "event": "end",
                "time": ended,
                "exit_code": result.returncode,
            },
        )
        if result.returncode != 0:
            raise RuntimeError(f"matrix run failed: {spec.run_id} (exit {result.returncode})")
        completed.append(spec.run_id)
    return completed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants = VARIANTS if args.variants == ["all"] else tuple(args.variants)
    specs = expand_matrix(
        datasets=tuple(args.datasets),
        variants=variants,
        seeds=tuple(args.seeds),
    )
    completed = run_matrix(
        specs,
        artifact_root=args.artifact_root,
        data_root=args.data_root,
        config_dir=args.config_dir,
        epochs_override=args.epochs,
        device=args.device,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    print("\n".join(completed))


if __name__ == "__main__":
    main()
