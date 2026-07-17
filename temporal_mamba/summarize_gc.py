"""Strict paired statistics and preregistered decision for the GC matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import GC_MATRIX_VARIANTS, load_experiment_config
from .run_gc_matrix import (
    GC_CONFIG_NAMES,
    GC_DATASETS,
    GC_STAGES,
    Stage,
    expand_gc_matrix,
)
from .run_matrix import RunSpec
from .summarize import _atomic_text, _check_numeric_tree
from .train import _canonical_hash, _config_payload


_T95_N5 = 2.776445105
_SYNTHETIC_VIEWS = (
    "val",
    "test",
    "length_256",
    "length_512",
    "parameter_ood",
    "noise_ood",
)
_SYNTHETIC_PRIMARY = ("length_512", "parameter_ood", "noise_ood")
_UCI_VIEWS = ("val", "test", "prefix50", "noise_025")
_UCI_PRIMARY = ("prefix50", "noise_025")
_GC_ORDERS = {
    "vanilla": 0,
    "two_pass": 0,
    "gc_k1": 1,
    "gc_k2": 2,
    "gc_k3": 3,
    "gc_k3_shuffled": 3,
    "gc_k3_noise": 3,
}
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
_GENERALIZED_SPLITS = {
    "train",
    "val",
    "test",
    "length_256",
    "length_512",
    "parameter_ood",
    "noise_ood",
}
_UCI_SPLITS = {"train", "val", "test"}
_UCI_OFFICIAL_SPLITS = {"train", "test"}


def _load_final(root: Path, spec: RunSpec) -> dict[str, Any]:
    path = root / spec.run_id / "final.json"
    if not path.exists():
        raise ValueError(f"missing final artifact: {spec.run_id}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid final artifact JSON: {spec.run_id}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"final artifact must be an object: {spec.run_id}")
    return value


def _load_sidecar(path: Path, spec: RunSpec) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing {path.name} sidecar: {spec.run_id}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {path.name} sidecar: {spec.run_id}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{spec.run_id} {path.name} sidecar must be an object")
    return value


def _require_hash(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _views_and_metric(dataset: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if dataset == "generalized_dynamics":
        return _SYNTHETIC_VIEWS, _SYNTHETIC_PRIMARY, "balanced_accuracy"
    if dataset == "uci_har":
        return _UCI_VIEWS, _UCI_PRIMARY, "macro_f1"
    raise ValueError(f"unsupported GC dataset: {dataset!r}")


def _score(metric: Mapping[str, Any], name: str, path: str) -> float:
    value = metric.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value at {path}.{name}: {value!r}")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path}.{name} must be within [0, 1]")
    return result


def _validate_dt(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    try:
        dt_min = float(value["dt_min"])
        dt_max = float(value["dt_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} requires numeric dt_min and dt_max") from exc
    if not math.isfinite(dt_min) or dt_min < 1e-3:
        raise ValueError(f"dt_min below bound at {path}.dt_min")
    if not math.isfinite(dt_max) or dt_max > 1e-1:
        raise ValueError(f"dt_max above bound at {path}.dt_max")
    if dt_min > dt_max:
        raise ValueError(f"dt_min exceeds dt_max at {path}")
    return dt_min, dt_max


def _validate_final(final: Mapping[str, Any], spec: RunSpec) -> None:
    expected_identity: tuple[tuple[str, object], ...] = (
        ("schema_version", 3),
        ("status", "complete"),
        ("run_id", spec.run_id),
        ("dataset", spec.dataset),
        ("variant", spec.variant),
        ("seed", spec.seed),
        ("gc_order", _GC_ORDERS[spec.variant]),
    )
    for field, expected in expected_identity:
        if final.get(field) != expected:
            raise ValueError(
                f"{spec.run_id} {field} mismatch: expected {expected!r}, "
                f"got {final.get(field)!r}"
            )

    commit = _require_hash(final.get("git_commit"), f"{spec.run_id}.git_commit")
    config_hash = _require_hash(final.get("config_hash"), f"{spec.run_id}.config_hash")
    manifest_hash = _require_hash(
        final.get("dataset_manifest_hash"),
        f"{spec.run_id}.dataset_manifest_hash",
    )
    hashes = final.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError(f"{spec.run_id} hashes must be an object")
    for name, expected in (
        ("git_commit", commit),
        ("config_sha256", config_hash),
        ("manifest_sha256", manifest_hash),
    ):
        if hashes.get(name) != expected:
            raise ValueError(f"{spec.run_id} hashes.{name} mismatch")
    environment = final.get("environment")
    if not isinstance(environment, Mapping) or environment.get("git_commit") != commit:
        raise ValueError(f"{spec.run_id} environment git_commit mismatch")

    parameter_count = final.get("parameter_count")
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise ValueError(f"{spec.run_id} parameter_count must be a positive integer")
    for field in ("per_order_auxiliary_losses", "per_order_error_rms"):
        value = final.get(field)
        if not isinstance(value, Mapping) or set(value) != {"order_0", "order_1", "order_2"}:
            raise ValueError(f"{spec.run_id} {field} orders mismatch")

    views, _, metric_name = _views_and_metric(spec.dataset)
    metrics = final.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(views):
        raise ValueError(f"{spec.run_id} metrics views mismatch")
    dt_diagnostics = final.get("dt_diagnostics")
    if not isinstance(dt_diagnostics, Mapping) or set(dt_diagnostics) != set(views):
        raise ValueError(f"{spec.run_id} dt_diagnostics views mismatch")
    for view in views:
        metric = metrics[view]
        if not isinstance(metric, Mapping):
            raise ValueError(f"{spec.run_id}.metrics.{view} must be an object")
        _score(metric, metric_name, f"{spec.run_id}.metrics.{view}")
        diagnostics = metric.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("finite") is not True:
            raise ValueError(f"finite flag is false at {spec.run_id}.metrics.{view}.diagnostics")
        metric_dt = _validate_dt(diagnostics, f"{spec.run_id}.metrics.{view}.diagnostics")
        artifact_dt = _validate_dt(
            dt_diagnostics[view], f"{spec.run_id}.dt_diagnostics.{view}"
        )
        if metric_dt != artifact_dt:
            raise ValueError(f"{spec.run_id} {view} dt_diagnostics mismatch")
    _check_numeric_tree(final, spec.run_id)


def _validate_sidecars(
    root: Path,
    spec: RunSpec,
    final: Mapping[str, Any],
    stage: Stage,
) -> tuple[str, str]:
    run_dir = root / spec.run_id
    config = _load_sidecar(run_dir / "config.json", spec)
    expected_config = load_experiment_config(
        _CONFIG_ROOT / GC_CONFIG_NAMES[spec.dataset],
        variant=spec.variant,
        seed=spec.seed,
    )
    if stage == "smoke":
        expected_config = replace(
            expected_config,
            training=replace(expected_config.training, epochs=1),
        )
    expected_config_payload = _config_payload(expected_config)
    if config != expected_config_payload:
        raise ValueError(f"{spec.run_id} sidecar does not match preregistered config")
    config_hash = _canonical_hash(config)
    expected_config_hash = _canonical_hash(expected_config_payload)
    if config_hash != expected_config_hash:
        raise ValueError(f"{spec.run_id} preregistered config hash mismatch")
    if config_hash != final["config_hash"]:
        raise ValueError(f"{spec.run_id} config sidecar hash mismatch")
    if config_hash != final["hashes"]["config_sha256"]:
        raise ValueError(f"{spec.run_id} config sidecar hash copy mismatch")

    manifest = _load_sidecar(run_dir / "dataset_manifest.json", spec)
    claimed_manifest_hash = _require_hash(
        manifest.get("manifest_sha256"),
        f"{spec.run_id}.dataset_manifest.json.manifest_sha256",
    )
    canonical_manifest = dict(manifest)
    canonical_manifest.pop("manifest_sha256", None)
    actual_manifest_hash = _canonical_hash(canonical_manifest)
    if claimed_manifest_hash != actual_manifest_hash:
        raise ValueError(f"{spec.run_id} manifest claim does not match canonical contents")
    _validate_manifest_contract(manifest, spec, expected_config_payload)
    if claimed_manifest_hash != final["dataset_manifest_hash"]:
        raise ValueError(f"{spec.run_id} manifest sidecar hash mismatch")
    if claimed_manifest_hash != final["hashes"]["manifest_sha256"]:
        raise ValueError(f"{spec.run_id} manifest sidecar hash copy mismatch")
    return config_hash, claimed_manifest_hash


def _validate_manifest_contract(
    manifest: Mapping[str, Any],
    spec: RunSpec,
    expected_config: Mapping[str, Any],
) -> None:
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError(f"{spec.run_id} manifest schema_version must be 1")
    data_seed = manifest.get("data_seed")
    if isinstance(data_seed, bool) or data_seed != expected_config["data_seed"]:
        raise ValueError(f"{spec.run_id} manifest data_seed mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{spec.run_id} manifest files must be an object")
    if spec.dataset == "generalized_dynamics":
        generator_version = manifest.get("generator_version")
        if not isinstance(generator_version, str) or not generator_version:
            raise ValueError(f"{spec.run_id} generalized manifest generator_version missing")
        splits = manifest.get("splits")
        sizes = manifest.get("sizes")
        if not isinstance(splits, list) or set(splits) != _GENERALIZED_SPLITS:
            raise ValueError(f"{spec.run_id} generalized manifest splits mismatch")
        if not isinstance(sizes, Mapping) or set(sizes) != _GENERALIZED_SPLITS:
            raise ValueError(f"{spec.run_id} generalized manifest sizes mismatch")
        if set(files) != _GENERALIZED_SPLITS:
            raise ValueError(f"{spec.run_id} generalized manifest files mismatch")
        return
    source_hash = manifest.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise ValueError(f"{spec.run_id} UCI manifest source hash missing")
    if set(files) != _UCI_SPLITS:
        raise ValueError(f"{spec.run_id} UCI manifest files mismatch")
    official_shapes = manifest.get("official_shapes")
    if not isinstance(official_shapes, Mapping) or set(official_shapes) != _UCI_OFFICIAL_SPLITS:
        raise ValueError(f"{spec.run_id} UCI manifest official_shapes mismatch")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, Mapping) or set(subjects) != _UCI_SPLITS:
        raise ValueError(f"{spec.run_id} UCI manifest subjects mismatch")
    signal_names = manifest.get("signal_names")
    if not isinstance(signal_names, list) or len(signal_names) != expected_config["signal_dim"]:
        raise ValueError(f"{spec.run_id} UCI manifest signal_names mismatch")


def _validate_matrix(
    root: Path,
    specs: Sequence[RunSpec],
    stage: Stage,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_ids = {spec.run_id for spec in specs}
    discovered_ids = {
        path.parent.name
        for dataset in GC_DATASETS
        for path in root.glob(f"{dataset}-*-seed*/final.json")
    }
    unexpected = sorted(discovered_ids - expected_ids)
    if unexpected:
        raise ValueError(f"unexpected GC final artifacts: {unexpected}")

    finals: dict[str, dict[str, Any]] = {}
    commits: set[str] = set()
    manifests: dict[str, set[str]] = {dataset: set() for dataset in GC_DATASETS}
    config_hashes: dict[str, set[str]] = {dataset: set() for dataset in GC_DATASETS}
    parameter_counts: dict[str, set[int]] = {dataset: set() for dataset in GC_DATASETS}
    for spec in specs:
        final = _load_final(root, spec)
        _validate_final(final, spec)
        config_hash, manifest_hash = _validate_sidecars(root, spec, final, stage)
        commits.add(str(final["git_commit"]))
        config_hashes[spec.dataset].add(config_hash)
        manifests[spec.dataset].add(manifest_hash)
        parameter_counts[spec.dataset].add(int(final["parameter_count"]))
        finals[spec.run_id] = final
    if len(commits) != 1:
        raise ValueError(f"git_commit mismatch across GC matrix: {sorted(commits)}")
    for dataset in GC_DATASETS:
        if len(manifests[dataset]) != 1:
            raise ValueError(
                f"dataset manifest mismatch for {dataset}: {sorted(manifests[dataset])}"
            )
        if len(parameter_counts[dataset]) != 1:
            raise ValueError(
                f"parameter_count mismatch for {dataset}: {sorted(parameter_counts[dataset])}"
            )
    validation = {
        "expected_runs": len(specs),
        "complete_runs": len(finals),
        "git_commit": next(iter(commits)),
        "dataset_manifest_hashes": {
            dataset: next(iter(manifests[dataset])) for dataset in GC_DATASETS
        },
        "config_hashes": {
            dataset: sorted(config_hashes[dataset]) for dataset in GC_DATASETS
        },
        "parameter_counts": {
            dataset: next(iter(parameter_counts[dataset])) for dataset in GC_DATASETS
        },
        "provenance_valid": True,
        "numerics_valid": True,
    }
    return validation, finals


def _paired_stats(values: Sequence[float], *, confirm: bool) -> dict[str, Any]:
    if not values:
        raise ValueError("paired statistics require at least one seed")
    numeric = [float(value) for value in values]
    sample_std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    result: dict[str, Any] = {
        "n": len(numeric),
        "values": numeric,
        "mean": statistics.fmean(numeric),
        "sample_std": sample_std,
        "ci95": None,
    }
    if confirm:
        if len(numeric) != 5:
            raise ValueError("confirm paired statistics require exactly five seeds")
        margin = _T95_N5 * sample_std / math.sqrt(5)
        result["ci95"] = {
            "critical_value": _T95_N5,
            "lower": result["mean"] - margin,
            "upper": result["mean"] + margin,
            "margin": margin,
        }
    return result


def _domain_summary(
    dataset: str,
    specs: Sequence[RunSpec],
    finals: Mapping[str, Mapping[str, Any]],
    *,
    confirm: bool,
) -> dict[str, Any]:
    _, primary_views, metric_name = _views_and_metric(dataset)
    seeds = sorted({spec.seed for spec in specs if spec.dataset == dataset})
    values: dict[str, dict[int, dict[str, Any]]] = {
        variant: {} for variant in GC_MATRIX_VARIANTS
    }
    for variant in GC_MATRIX_VARIANTS:
        for seed in seeds:
            run_id = f"{dataset}-{variant}-seed{seed}"
            final = finals[run_id]
            view_scores = {
                view: _score(
                    final["metrics"][view],
                    metric_name,
                    f"{run_id}.metrics.{view}",
                )
                for view in primary_views
            }
            values[variant][seed] = {
                "id": _score(
                    final["metrics"]["test"],
                    metric_name,
                    f"{run_id}.metrics.test",
                ),
                "primary_ood": statistics.fmean(view_scores.values()),
                "primary_views": view_scores,
            }

    def paired(upper: str, lower: str, field: str) -> dict[str, Any]:
        return _paired_stats(
            [values[upper][seed][field] - values[lower][seed][field] for seed in seeds],
            confirm=confirm,
        )

    effect = paired("gc_k3", "two_pass", "primary_ood")
    id_effect = paired("gc_k3", "two_pass", "id")
    shuffled = paired("gc_k3", "gc_k3_shuffled", "primary_ood")
    noise = paired("gc_k3", "gc_k3_noise", "primary_ood")
    threshold = 0.02 if dataset == "generalized_dynamics" else 0.01
    comparison_tolerance = 1e-12
    threshold_met = effect["mean"] >= threshold - comparison_tolerance
    ci_lower_positive = bool(confirm and effect["ci95"]["lower"] > 0.0)
    controls_won = shuffled["mean"] > 0.0 and noise["mean"] > 0.0
    id_degradation_ok = id_effect["mean"] >= -0.01 - comparison_tolerance
    return {
        "metric": metric_name,
        "primary_ood_views": list(primary_views),
        "threshold": threshold,
        "per_seed": {
            variant: {str(seed): values[variant][seed] for seed in seeds}
            for variant in GC_MATRIX_VARIANTS
        },
        "k3_minus_two_pass": effect,
        "id_k3_minus_two_pass": id_effect,
        "controls": {
            "k3_minus_gc_k3_shuffled": shuffled,
            "k3_minus_gc_k3_noise": noise,
            "passed": controls_won,
        },
        "criteria": {
            "mean_threshold_met": threshold_met,
            "ci_lower_positive": ci_lower_positive,
            "controls_won": controls_won,
            "id_degradation_within_0_01": id_degradation_ok,
            "provenance_and_numerics_valid": True,
            "full_pass": bool(
                confirm
                and threshold_met
                and ci_lower_positive
                and controls_won
                and id_degradation_ok
            ),
        },
    }


def _screen_gate(domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"domains": {}}
    for dataset, domain in domains.items():
        per_seed = domain["per_seed"]
        seeds = sorted(per_seed["gc_k3"], key=int)
        simultaneous_wins = [
            int(seed)
            for seed in seeds
            if per_seed["gc_k3"][seed]["primary_ood"]
            > per_seed["gc_k3_shuffled"][seed]["primary_ood"]
            and per_seed["gc_k3"][seed]["primary_ood"]
            > per_seed["gc_k3_noise"][seed]["primary_ood"]
        ]
        mean_positive = domain["k3_minus_two_pass"]["mean"] > 0.0
        result["domains"][dataset] = {
            "k3_minus_two_pass_mean_positive": mean_positive,
            "simultaneous_control_win_seeds": simultaneous_wins,
            "simultaneous_control_win_count": len(simultaneous_wins),
            "passed": mean_positive and len(simultaneous_wins) >= 2,
        }
    result["at_least_one_domain_passed"] = any(
        value["passed"] for value in result["domains"].values()
    )
    return result


def _confirm_decision(domains: Mapping[str, Mapping[str, Any]]) -> str:
    criteria = [domain["criteria"] for domain in domains.values()]
    if not all(item["controls_won"] for item in criteria):
        return "not_supported"
    full_passes = sum(bool(item["full_pass"]) for item in criteria)
    if full_passes == 2:
        return "supported"
    if full_passes == 1:
        return "uncertain"
    if any(item["mean_threshold_met"] and not item["ci_lower_positive"] for item in criteria):
        return "uncertain"
    return "not_supported"


def _report_zh(summary: Mapping[str, Any]) -> str:
    validation = summary["validation"]
    lines = [
        "# 广义坐标最小消融报告",
        "",
        f"- 阶段：`{summary['stage']}`",
        f"- 作业完整性：{validation['complete_runs']}/{validation['expected_runs']}",
        f"- 数值检查：{'通过' if validation['numerics_valid'] else '失败'}",
        "",
        "## 溯源",
        "",
        f"- Git 提交：`{validation['git_commit']}`",
        f"- 数据 manifest：`{json.dumps(validation['dataset_manifest_hashes'], ensure_ascii=False, sort_keys=True)}`",
        f"- 参数量：`{json.dumps(validation['parameter_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    for dataset in GC_DATASETS:
        hashes = validation["config_hashes"][dataset]
        lines.append(f"- {dataset} config hashes（{len(hashes)}）：`{json.dumps(hashes)}`")
    lines.append("")
    for dataset in GC_DATASETS:
        domain = summary["domains"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"主 OOD 指标：`{domain['metric']}`；视图：`{', '.join(domain['primary_ood_views'])}`。",
                "",
                "### 逐种子 ID/OOD 值",
                "",
                "| 变体 | 种子 | ID | 主 OOD |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in GC_MATRIX_VARIANTS:
            for seed, value in domain["per_seed"][variant].items():
                lines.append(
                    f"| {variant} | {seed} | {value['id']:.6f} | {value['primary_ood']:.6f} |"
                )
        effect = domain["k3_minus_two_pass"]
        ci = effect["ci95"]
        ci_text = (
            "不适用"
            if ci is None
            else f"[{ci['lower']:+.6f}, {ci['upper']:+.6f}]"
        )
        lines.extend(
            [
                "",
                "### 配对效应与控制",
                "",
                f"- K3 - two_pass 主 OOD：{effect['mean']:+.6f}，样本标准差 {effect['sample_std']:.6f}，95% CI {ci_text}",
                f"- K3 - two_pass ID：{domain['id_k3_minus_two_pass']['mean']:+.6f}",
                f"- K3 - shuffled：{domain['controls']['k3_minus_gc_k3_shuffled']['mean']:+.6f}",
                f"- K3 - noise：{domain['controls']['k3_minus_gc_k3_noise']['mean']:+.6f}",
                f"- 完整门槛：{'通过' if domain['criteria']['full_pass'] else '未通过'}",
                "",
            ]
        )
    if summary["screen_gate"] is not None:
        gate = summary["screen_gate"]
        lines.extend(
            [
                "## 初筛门控",
                "",
                f"- 至少一个数据域通过：{gate['at_least_one_domain_passed']}",
            ]
        )
        for dataset, value in gate["domains"].items():
            lines.append(
                f"- {dataset}：passed={value['passed']}，同时胜双控制的种子数={value['simultaneous_control_win_count']}"
            )
        lines.append("")
    lines.extend(["## 决策", "", f"**结论：{summary['decision']}**", ""])
    return "\n".join(lines)


def summarize_gc_matrix(artifact_root: str | Path, stage: Stage) -> dict[str, object]:
    """Validate one exact stage and emit its paired, preregistered decision."""

    specs = expand_gc_matrix(stage)
    root = Path(artifact_root)
    validation, finals = _validate_matrix(root, specs, stage)
    domains = {
        dataset: _domain_summary(
            dataset,
            specs,
            finals,
            confirm=stage == "confirm",
        )
        for dataset in GC_DATASETS
    }
    screen_gate = _screen_gate(domains) if stage == "screen" else None
    if stage == "confirm":
        decision = _confirm_decision(domains)
    elif stage == "screen":
        decision = "uncertain" if screen_gate["at_least_one_domain_passed"] else "not_supported"
    else:
        decision = "uncertain"
    summary: dict[str, Any] = {
        "schema_version": 3,
        "stage": stage,
        "completed_jobs": validation["complete_runs"],
        "git_commit": validation["git_commit"],
        "dataset_manifest_hashes": dict(validation["dataset_manifest_hashes"]),
        "validation": validation,
        "domains": domains,
        "screen_gate": screen_gate,
        "decision": decision,
    }
    _atomic_text(
        root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )
    _atomic_text(root / "report_zh.md", _report_zh(summary))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=GC_STAGES)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_gc_matrix(args.artifact_root, args.stage)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
