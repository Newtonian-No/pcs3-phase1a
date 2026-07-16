"""Deterministic, guarded training engine shared by both temporal datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_experiment_config
from .datasets.temporal_logic import TemporalLogicDataset, build_temporal_logic_manifest
from .datasets.uci_har import UCIHARDataset, prepare_uci_har
from .losses import compute_total_loss
from .metrics import binary_metrics, multiclass_metrics
from .model import TemporalMambaModel
from .numerics import (
    NumericalFailure,
    assert_finite_model,
    assert_finite_tensor,
    write_failure_artifact,
)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_datasets(config: ExperimentConfig, data_root: str | Path) -> dict[str, Dataset]:
    data_root = Path(data_root)
    if config.dataset == "temporal_logic":
        manifest_path = data_root / "manifest.json"
        if not manifest_path.exists():
            build_temporal_logic_manifest(
                data_root,
                {
                    "train": config.data.train_size,
                    "val": config.data.val_size,
                    "test": config.data.test_size,
                    "long_test": config.data.long_test_size,
                },
                data_seed=config.data_seed,
                event_dim=config.signal_dim,
                seq_len=config.seq_len,
                long_seq_len=2 * config.seq_len,
            )
        return {
            split: TemporalLogicDataset(data_root, split, transform=config.time_transform)
            for split in ("train", "val", "test", "long_test")
        }
    if config.dataset == "uci_har":
        if not (data_root / "manifest.json").exists():
            prepare_uci_har(data_root, data_seed=config.data_seed)
        return {
            split: UCIHARDataset(data_root, split, transform=config.time_transform)
            for split in ("train", "val", "test")
        }
    raise ValueError(f"unknown dataset: {config.dataset}")


def build_loaders(
    datasets: Mapping[str, Dataset],
    config: ExperimentConfig,
    *,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], torch.Generator]:
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            worker_init_fn=_seed_worker,
            persistent_workers=num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    return loaders, generator


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
    signal = batch["signal"].to(device=device, dtype=torch.float32, non_blocking=True)
    target = batch["target"].to(device=device, non_blocking=True)
    return features, signal, target


def _predictions(logits: Tensor, dataset: str) -> Tensor:
    if dataset == "temporal_logic":
        return (logits[:, 0] >= 0).long()
    return logits.argmax(dim=-1)


def _classification_metrics(dataset: str, target: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    if dataset == "temporal_logic":
        return binary_metrics(target, predicted)
    return multiclass_metrics(target, predicted, num_classes=6)


def _guard_output(output) -> None:
    assert_finite_tensor("logits", output.logits)
    if output.position_error is not None:
        assert_finite_tensor("position_error", output.position_error)
    if output.velocity_error is not None:
        assert_finite_tensor("velocity_error", output.velocity_error)
    for name, value in output.diagnostics.items():
        if isinstance(value, Tensor):
            assert_finite_tensor(f"diagnostics.{name}", value)
    finite = output.diagnostics.get("finite")
    if isinstance(finite, Tensor) and not bool(finite):
        raise NumericalFailure("diagnostics.finite", observed_min=None, observed_max=None)


def train_epoch(
    model: TemporalMambaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    config: ExperimentConfig,
    *,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_auxiliary = 0.0
    sample_count = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    diagnostic_min = math.inf
    diagnostic_max = -math.inf
    last_output = None
    last_weight = 0.0

    for batch in loader:
        features, signal, target = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            features,
            signal,
            variant=config.variant,
            return_diagnostics=True,
        )
        _guard_output(output)
        breakdown = compute_total_loss(
            output,
            target,
            dataset=config.dataset,
            variant=config.variant,
            epoch=epoch,
            total_epochs=config.training.epochs,
            lambda_aux=config.training.lambda_aux,
            aux_warmup_fraction=config.training.aux_warmup_fraction,
        )
        assert_finite_tensor("loss.total", breakdown.total)
        breakdown.total.backward()
        assert_finite_model(model, check_gradients=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        assert_finite_model(model, check_gradients=False)

        batch_size = int(target.shape[0])
        sample_count += batch_size
        total_loss += float(breakdown.total.detach()) * batch_size
        total_task += float(breakdown.task.detach()) * batch_size
        total_auxiliary += float(breakdown.auxiliary.detach()) * batch_size
        targets.append(target.detach().cpu().numpy().reshape(-1))
        predictions.append(_predictions(output.logits.detach(), config.dataset).cpu().numpy())
        diagnostic_min = min(diagnostic_min, float(output.diagnostics["dt_min"].detach()))
        diagnostic_max = max(diagnostic_max, float(output.diagnostics["dt_max"].detach()))
        global_step += 1
        last_output = output
        last_weight = breakdown.aux_weight

    if sample_count == 0 or last_output is None:
        raise ValueError("training loader is empty")
    metrics = _classification_metrics(
        config.dataset,
        np.concatenate(targets).astype(np.int64),
        np.concatenate(predictions).astype(np.int64),
    )
    metrics.update(
        {
            "loss": total_loss / sample_count,
            "task_loss": total_task / sample_count,
            "auxiliary_loss": total_auxiliary / sample_count,
            "aux_weight": last_weight,
            "pass_count": last_output.pass_count,
            "uses_error": last_output.uses_error,
            "dt_min": max(diagnostic_min, config.model.dt_min),
            "dt_max": min(diagnostic_max, config.model.dt_max),
            "global_step": global_step,
            "finite": True,
        }
    )
    return metrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    sample_count = 0
    total_loss = 0.0
    targets: list[np.ndarray] = []
    base_targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    families: list[str] = []
    dt_min = math.inf
    dt_max = -math.inf
    error_rms = 0.0
    error_max = 0.0
    output_rms = 0.0
    output_max = 0.0
    finite = True

    with torch.no_grad():
        for batch in loader:
            features, signal, target = _move_batch(batch, device)
            output = model(
                features,
                signal,
                variant=config.variant,
                return_diagnostics=True,
            )
            _guard_output(output)
            task_loss = compute_total_loss(
                output,
                target,
                dataset=config.dataset,
                variant=config.variant,
                epoch=config.training.epochs,
                total_epochs=config.training.epochs,
                lambda_aux=0.0,
                aux_warmup_fraction=config.training.aux_warmup_fraction,
            ).task
            batch_size = int(target.shape[0])
            sample_count += batch_size
            total_loss += float(task_loss) * batch_size
            targets.append(target.cpu().numpy().reshape(-1))
            base = batch.get("base_target", target)
            if isinstance(base, Tensor):
                base_targets.append(base.cpu().numpy().reshape(-1))
            else:
                base_targets.append(np.asarray(base).reshape(-1))
            predictions.append(_predictions(output.logits, config.dataset).cpu().numpy())
            batch_families = batch.get("formula_family", [""] * batch_size)
            families.extend([str(item) for item in batch_families])
            diagnostics = output.diagnostics
            dt_min = min(dt_min, float(diagnostics["dt_min"]))
            dt_max = max(dt_max, float(diagnostics["dt_max"]))
            error_rms = max(error_rms, float(diagnostics["error_rms"]))
            error_max = max(error_max, float(diagnostics["error_max"]))
            output_rms = max(output_rms, float(diagnostics["output_rms"]))
            output_max = max(output_max, float(diagnostics["output_max"]))
            finite = finite and bool(diagnostics["finite"])

    if sample_count == 0:
        raise ValueError("evaluation loader is empty")
    target_array = np.concatenate(targets).astype(np.int64)
    base_array = np.concatenate(base_targets).astype(np.int64)
    predicted_array = np.concatenate(predictions).astype(np.int64)
    metrics = _classification_metrics(config.dataset, target_array, predicted_array)
    metrics["loss"] = total_loss / sample_count
    metrics["frozen_label_metrics"] = _classification_metrics(
        config.dataset,
        base_array,
        predicted_array,
    )
    per_family: dict[str, dict[str, object]] = {}
    if config.dataset == "temporal_logic":
        family_array = np.asarray(families)
        for family in sorted(set(families)):
            if not family:
                continue
            mask = family_array == family
            per_family[family] = binary_metrics(target_array[mask], predicted_array[mask])
    metrics["per_family"] = per_family
    metrics["diagnostics"] = {
        "dt_min": max(dt_min, config.model.dt_min),
        "dt_max": min(dt_max, config.model.dt_max),
        "error_rms": error_rms,
        "error_max": error_max,
        "output_rms": output_rms,
        "output_max": output_max,
        "finite": finite,
    }
    return metrics


def overfit_tiny_batch(
    model: TemporalMambaModel,
    batch: Mapping[str, Any],
    config: ExperimentConfig,
    *,
    device: torch.device,
    max_steps: int = 500,
    target_accuracy: float = 0.98,
) -> dict[str, Any]:
    if max_steps <= 0 or not 0 < target_accuracy <= 1:
        raise ValueError("invalid tiny-batch gate settings")
    features, signal, target = _move_batch(batch, device)
    model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(config.training.lr, 5e-3),
        weight_decay=0.0,
    )
    final_accuracy = 0.0
    final_loss = math.inf
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(features, signal, variant=config.variant, return_diagnostics=True)
        _guard_output(output)
        breakdown = compute_total_loss(
            output,
            target,
            dataset=config.dataset,
            variant=config.variant,
            epoch=config.training.epochs,
            total_epochs=config.training.epochs,
            lambda_aux=config.training.lambda_aux,
            aux_warmup_fraction=config.training.aux_warmup_fraction,
        )
        assert_finite_tensor("tiny.loss", breakdown.total)
        breakdown.total.backward()
        assert_finite_model(model, check_gradients=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        prediction = _predictions(output.logits.detach(), config.dataset)
        final_accuracy = float((prediction == target.long().reshape(-1)).float().mean())
        final_loss = float(breakdown.total.detach())
        if final_accuracy >= target_accuracy:
            return {
                "passed": True,
                "steps": step,
                "accuracy": final_accuracy,
                "loss": final_loss,
                "finite": True,
            }
    return {
        "passed": False,
        "steps": max_steps,
        "accuracy": final_accuracy,
        "loss": final_loss,
        "finite": True,
    }


def validation_selection_score(validation_metrics: Mapping[str, Any]) -> float:
    if "accuracy" not in validation_metrics:
        raise ValueError("validation metrics must contain accuracy")
    return float(validation_metrics["accuracy"])


def _input_dim(config: ExperimentConfig) -> int:
    if config.dataset == "temporal_logic":
        return config.signal_dim + 1 + 6 + 2 * config.signal_dim + 3
    return config.signal_dim + 1


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "pass_count": config.pass_count,
            "uses_error": config.uses_error,
            "uses_aux": config.uses_aux,
            "time_transform": config.time_transform,
        }
    )
    return payload


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_history(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _environment(device: torch.device, git_commit: str) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "git_commit": git_commit,
        "amp": False,
    }


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
):
    warmup_steps = int(math.ceil(total_steps * warmup_fraction))

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def run_training(
    *,
    config_path: str | Path,
    variant: str,
    seed: int,
    data_root: str | Path,
    artifact_root: str | Path,
    device: str | torch.device | None = None,
    epochs_override: int | None = None,
    resume: bool = True,
    num_workers: int = 0,
    overfit_only: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(config_path, variant=variant, seed=seed)
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("epochs_override must be positive")
        config = replace(config, training=replace(config.training, epochs=epochs_override))
    set_training_seed(seed)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    datasets = build_datasets(config, data_root)
    loaders, loader_generator = build_loaders(datasets, config, num_workers=num_workers)
    model = TemporalMambaModel(
        input_dim=_input_dim(config),
        signal_dim=config.signal_dim,
        num_outputs=config.num_outputs,
        model_config=config.model,
    ).to(resolved_device)

    run_id = f"{config.dataset}-{variant}-seed{seed}"
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload(config)
    config_hash = _canonical_hash(config_payload)
    data_manifest = json.loads((Path(data_root) / "manifest.json").read_text(encoding="utf-8"))
    dataset_hash = str(data_manifest["manifest_sha256"])
    git_commit = _git_commit()

    final_path = run_dir / "final.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("config_hash") == config_hash
            and existing.get("dataset_manifest_hash") == dataset_hash
            and existing.get("git_commit") == git_commit
        ):
            return existing
        raise ValueError(f"existing final artifact metadata mismatch for {run_id}")

    _atomic_json(run_dir / "config.json", config_payload)
    _atomic_json(run_dir / "environment.json", _environment(resolved_device, git_commit))
    _atomic_json(run_dir / "dataset_manifest.json", data_manifest)

    if overfit_only:
        limit = 64 if config.dataset == "temporal_logic" else 48
        subset = Subset(datasets["train"], range(min(limit, len(datasets["train"]))))
        gate_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        gate = overfit_tiny_batch(
            model,
            next(iter(gate_loader)),
            config,
            device=resolved_device,
            max_steps=800,
            target_accuracy=0.98 if config.dataset == "temporal_logic" else 0.95,
        )
        result = {"status": "complete" if gate["passed"] else "failed", "run_id": run_id, **gate}
        _atomic_json(run_dir / "overfit.json", result)
        if not gate["passed"]:
            raise RuntimeError(f"tiny-batch overfit gate failed for {run_id}: {gate}")
        return result

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    total_steps = config.training.epochs * len(loaders["train"])
    scheduler = _make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=config.training.warmup_fraction,
    )
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.jsonl"
    start_epoch = 0
    global_step = 0
    best_metric = -math.inf
    best_epoch = -1
    history_cursor = 0
    epochs_without_improvement = 0

    if resume and last_path.exists():
        checkpoint = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader_generator=loader_generator,
            map_location=resolved_device,
            expected_config_hash=config_hash,
            expected_dataset_manifest_hash=dataset_hash,
            expected_git_commit=git_commit,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        best_metric = float(checkpoint["best_metric"])
        history_cursor = int(checkpoint["history_cursor"])
        best_epoch = int(checkpoint.get("extra", {}).get("best_epoch", -1))
        epochs_without_improvement = int(
            checkpoint.get("extra", {}).get("epochs_without_improvement", 0)
        )

    last_validation: dict[str, Any] | None = None
    try:
        for epoch in range(start_epoch, config.training.epochs):
            training_metrics = train_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                config,
                device=resolved_device,
                epoch=epoch,
                global_step=global_step,
            )
            global_step = int(training_metrics["global_step"])
            validation_metrics = evaluate(
                model,
                loaders["val"],
                config,
                device=resolved_device,
            )
            last_validation = validation_metrics
            score = validation_selection_score(validation_metrics)
            improved = score > best_metric
            if improved:
                best_metric = score
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history_record = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "train": training_metrics,
                "val": validation_metrics,
                "selection_score": score,
                "improved": improved,
            }
            _append_history(history_path, history_record)
            history_cursor += 1
            checkpoint_kwargs = dict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                loader_generator=loader_generator,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                history_cursor=history_cursor,
                config_hash=config_hash,
                git_commit=git_commit,
                dataset_manifest_hash=dataset_hash,
                extra={
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
            if improved:
                save_checkpoint(best_path, **checkpoint_kwargs)
            save_checkpoint(last_path, **checkpoint_kwargs)
            if epochs_without_improvement >= config.training.patience:
                break
    except Exception as error:
        component = error.component if isinstance(error, NumericalFailure) else "training"
        write_failure_artifact(
            run_dir / "failure.json",
            run_id=run_id,
            epoch=max(start_epoch, 0),
            batch=-1,
            tensor_name=component,
            error=error,
            last_healthy_checkpoint=last_path if last_path.exists() else None,
        )
        raise

    if not best_path.exists() or last_validation is None:
        raise RuntimeError(f"no completed training epoch for {run_id}")
    load_checkpoint(
        best_path,
        model=model,
        map_location=resolved_device,
        expected_config_hash=config_hash,
        expected_dataset_manifest_hash=dataset_hash,
        expected_git_commit=git_commit,
        restore_rng=False,
    )
    validation_metrics = evaluate(model, loaders["val"], config, device=resolved_device)
    test_metrics = evaluate(model, loaders["test"], config, device=resolved_device)
    long_metrics = (
        evaluate(model, loaders["long_test"], config, device=resolved_device)
        if "long_test" in loaders
        else None
    )
    original_test_metrics = None
    if config.time_transform != "none":
        if config.dataset == "temporal_logic":
            original_dataset = TemporalLogicDataset(data_root, "test", transform="none")
        else:
            original_dataset = UCIHARDataset(data_root, "test", transform="none")
        original_loader = DataLoader(
            original_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        original_test_metrics = evaluate(model, original_loader, config, device=resolved_device)

    final: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id,
        "dataset": config.dataset,
        "variant": config.variant,
        "seed": config.seed,
        "config_hash": config_hash,
        "dataset_manifest_hash": dataset_hash,
        "git_commit": git_commit,
        "selection_split": "val",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "completed_epochs": history_cursor,
        "global_step": global_step,
        "pass_count": config.pass_count,
        "uses_error": config.uses_error,
        "uses_aux": config.uses_aux,
        "time_transform": config.time_transform,
        "metrics": {
            "val": validation_metrics,
            "test": test_metrics,
            "long_test": long_metrics,
            "original_test": original_test_metrics,
        },
    }
    _atomic_json(final_path, final)
    return final


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overfit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = args.data_root or Path("data") / raw_config["dataset"]
    try:
        result = run_training(
            config_path=args.config,
            variant=args.variant,
            seed=args.seed,
            data_root=data_root,
            artifact_root=args.artifact_root,
            device=args.device,
            epochs_override=args.epochs,
            resume=not args.no_resume,
            num_workers=args.num_workers,
            overfit_only=args.overfit_only,
        )
    except Exception as error:
        run_id = f"{raw_config['dataset']}-{args.variant}-seed{args.seed}"
        run_dir = args.artifact_root / run_id
        failure_path = run_dir / "failure.json"
        if not failure_path.exists():
            component = error.component if isinstance(error, NumericalFailure) else "setup"
            write_failure_artifact(
                failure_path,
                run_id=run_id,
                epoch=-1,
                batch=-1,
                tensor_name=component,
                error=error,
                last_healthy_checkpoint=(
                    run_dir / "last.pt" if (run_dir / "last.pt").exists() else None
                ),
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
