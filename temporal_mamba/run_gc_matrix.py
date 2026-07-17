"""Dedicated, resumable launcher for the generalized-coordinate matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from .config import GC_CONFIRM_SEEDS, GC_MATRIX_VARIANTS, GC_SEEDS, load_experiment_config
from .run_matrix import RunSpec
from .train import _canonical_hash, _config_payload, _git_commit


GC_DATASETS = ("generalized_dynamics", "uci_har")
GC_CONFIG_NAMES = {
    "generalized_dynamics": "generalized_dynamics_gc.json",
    "uci_har": "uci_har_gc.json",
}
GC_STAGES = ("smoke", "screen", "confirm")
Stage = Literal["smoke", "screen", "confirm"]


def _stage_seeds(stage: Stage) -> tuple[int, ...]:
    if stage == "smoke":
        return (42,)
    if stage == "screen":
        return tuple(GC_SEEDS)
    if stage == "confirm":
        return tuple(GC_CONFIRM_SEEDS)
    raise ValueError(f"stage must be one of {GC_STAGES}, got {stage!r}")


def expand_gc_matrix(stage: Stage) -> tuple[RunSpec, ...]:
    """Expand exactly the preregistered GC jobs for ``stage``."""

    seeds = _stage_seeds(stage)
    return tuple(
        RunSpec(dataset, variant, seed)
        for dataset in GC_DATASETS
        for variant in GC_MATRIX_VARIANTS
        for seed in seeds
    )


def _append_status(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _config_path(config_dir: Path, spec: RunSpec) -> Path:
    try:
        return config_dir / GC_CONFIG_NAMES[spec.dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported GC dataset: {spec.dataset!r}") from exc


def _expected_metadata(
    spec: RunSpec,
    *,
    stage: Stage,
    config_dir: Path,
    data_root: Path,
) -> dict[str, str] | None:
    config_path = _config_path(config_dir, spec)
    manifest_path = data_root / spec.dataset / "manifest.json"
    if not config_path.exists() or not manifest_path.exists():
        return None
    config = load_experiment_config(config_path, variant=spec.variant, seed=spec.seed)
    if stage == "smoke":
        config = replace(config, training=replace(config.training, epochs=1))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = manifest["manifest_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid dataset manifest for {spec.dataset}: {exc}") from exc
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError(f"invalid dataset manifest hash for {spec.dataset}")
    return {
        "config_hash": _canonical_hash(_config_payload(config)),
        "dataset_manifest_hash": manifest_hash,
        "git_commit": _git_commit(),
    }


def _is_reusable_final(
    path: Path,
    spec: RunSpec,
    expected: Mapping[str, str] | None,
) -> bool:
    if not path.exists():
        return False
    try:
        final = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid completed artifact for {spec.run_id}: {exc}") from exc
    required = {
        "schema_version": 3,
        "status": "complete",
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "seed": spec.seed,
    }
    if expected is None or any(final.get(key) != value for key, value in required.items()):
        raise ValueError(f"completed artifact metadata mismatch for {spec.run_id}")
    if any(final.get(key) != value for key, value in expected.items()):
        raise ValueError(f"completed artifact metadata mismatch for {spec.run_id}")
    return True


def _training_command(
    spec: RunSpec,
    *,
    stage: Stage,
    artifact_root: Path,
    data_root: Path,
    config_dir: Path,
    device: str | None,
    num_workers: int,
    overfit_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "temporal_mamba.train",
        "--config",
        str(_config_path(config_dir, spec)),
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
    if stage == "smoke":
        command.extend(("--epochs", "1"))
    if overfit_only:
        command.append("--overfit-only")
    if device is not None:
        command.extend(("--device", device))
    return command


def _run_subprocess(
    command: Sequence[str],
    *,
    spec: RunSpec,
    event: str,
    status_path: Path,
    environment: Mapping[str, str],
) -> None:
    _append_status(
        status_path,
        {
            "run_id": spec.run_id,
            "event": f"{event}_start",
            "time": datetime.now(timezone.utc).isoformat(),
        },
    )
    result = subprocess.run(list(command), check=False, env=dict(environment))
    _append_status(
        status_path,
        {
            "run_id": spec.run_id,
            "event": f"{event}_end",
            "time": datetime.now(timezone.utc).isoformat(),
            "exit_code": result.returncode,
        },
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"GC matrix {event} failed: {spec.run_id} (exit {result.returncode})"
        )


def run_gc_matrix(
    stage: Stage,
    *,
    artifact_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path = "configs",
    specs: Iterable[RunSpec] | None = None,
    device: str | None = None,
    num_workers: int = 0,
    dry_run: bool = False,
) -> list[str]:
    """Run only the requested GC stage, rejecting unsafe artifact reuse."""

    allowed = expand_gc_matrix(stage)
    selected = list(allowed if specs is None else specs)
    unknown = [spec.run_id for spec in selected if spec not in set(allowed)]
    if unknown:
        raise ValueError(f"jobs do not belong to stage {stage}: {unknown}")
    if dry_run:
        return [spec.run_id for spec in selected]

    artifact_root = Path(artifact_root)
    data_root = Path(data_root)
    config_dir = Path(config_dir)
    status_path = artifact_root / "matrix_status.jsonl"
    pycache_root = Path(tempfile.gettempdir()) / f"pcs3-pycache-{_git_commit()[:12]}"
    completed: list[str] = []

    for spec in selected:
        expected = _expected_metadata(
            spec,
            stage=stage,
            config_dir=config_dir,
            data_root=data_root,
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

        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(pycache_root)
        environment["PYTHONHASHSEED"] = str(spec.seed)
        if stage == "smoke":
            tiny_command = _training_command(
                spec,
                stage=stage,
                artifact_root=artifact_root,
                data_root=data_root,
                config_dir=config_dir,
                device=device,
                num_workers=num_workers,
                overfit_only=True,
            )
            _run_subprocess(
                tiny_command,
                spec=spec,
                event="tiny",
                status_path=status_path,
                environment=environment,
            )
        command = _training_command(
            spec,
            stage=stage,
            artifact_root=artifact_root,
            data_root=data_root,
            config_dir=config_dir,
            device=device,
            num_workers=num_workers,
        )
        _run_subprocess(
            command,
            spec=spec,
            event="train",
            status_path=status_path,
            environment=environment,
        )
        completed.append(spec.run_id)
    return completed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=GC_STAGES)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    completed = run_gc_matrix(
        args.stage,
        artifact_root=args.artifact_root,
        data_root=args.data_root,
        config_dir=args.config_dir,
        device=args.device,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    print("\n".join(completed))


if __name__ == "__main__":
    main()
