"""Strict completeness validation and paired causal-ablation summaries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import VARIANTS
from .run_matrix import DATASETS, RunSpec, expand_matrix


_PAIRS = (
    ("two_pass", "vanilla"),
    ("error_inject", "two_pass"),
    ("error_aux", "error_inject"),
)


def _load_final(artifact_root: Path, spec: RunSpec) -> dict[str, Any]:
    path = artifact_root / spec.run_id / "final.json"
    if not path.exists():
        raise ValueError(f"missing final artifact: {spec.run_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid final artifact JSON: {spec.run_id}: {exc}") from exc


def _check_numeric_tree(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite value at {path}: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_numeric_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        if "finite" in value and value["finite"] is not True:
            raise ValueError(f"finite flag is false at {path}.finite")
        if "dt_min" in value and float(value["dt_min"]) < 1e-3:
            raise ValueError(f"dt_min below bound at {path}.dt_min")
        if "dt_max" in value and float(value["dt_max"]) > 1e-1:
            raise ValueError(f"dt_max above bound at {path}.dt_max")
        for key, item in value.items():
            _check_numeric_tree(item, f"{path}.{key}")


def validate_matrix(
    artifact_root: str | Path,
    specs: Sequence[RunSpec] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    specs = expand_matrix() if specs is None else list(specs)
    finals: dict[str, dict[str, Any]] = {}
    commits: set[str] = set()
    dataset_hashes: dict[str, set[str]] = {}
    for spec in specs:
        final = _load_final(artifact_root, spec)
        for field, expected in (
            ("run_id", spec.run_id),
            ("dataset", spec.dataset),
            ("variant", spec.variant),
            ("seed", spec.seed),
        ):
            if final.get(field) != expected:
                raise ValueError(
                    f"{spec.run_id} {field} mismatch: expected {expected!r}, got {final.get(field)!r}"
                )
        if final.get("status") != "complete":
            raise ValueError(f"run is not complete: {spec.run_id}")
        for required in ("config_hash", "dataset_manifest_hash", "git_commit", "metrics"):
            if required not in final:
                raise ValueError(f"{spec.run_id} missing metadata: {required}")
        _check_numeric_tree(final["metrics"], f"{spec.run_id}.metrics")
        commits.add(str(final["git_commit"]))
        dataset_hashes.setdefault(spec.dataset, set()).add(str(final["dataset_manifest_hash"]))
        finals[spec.run_id] = final
    if len(commits) != 1:
        raise ValueError(f"git_commit mismatch across matrix: {sorted(commits)}")
    for dataset, hashes in dataset_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"dataset_manifest_hash mismatch for {dataset}: {sorted(hashes)}")
    return {
        "expected_runs": len(specs),
        "complete_runs": len(finals),
        "git_commit": next(iter(commits)) if commits else None,
        "dataset_manifest_hashes": {
            dataset: next(iter(hashes)) for dataset, hashes in dataset_hashes.items()
        },
        "finals": finals,
    }


def _mean_std(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot aggregate an empty value set")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _report_markdown(summary: Mapping[str, Any], finals: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# Temporal Mamba causal ablation report",
        "",
        f"Verified runs: {summary['validation']['complete_runs']}/{summary['validation']['expected_runs']}",
        "",
    ]
    for dataset, variants in summary["groups"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Variant | Test accuracy mean | Sample std | Seeds |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            if variant not in variants:
                continue
            stats = variants[variant]
            lines.append(
                f"| {variant} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['n']} |"
            )
        lines.extend(["", "Paired causal deltas:", ""])
        for name, stats in summary["paired_deltas"].get(dataset, {}).items():
            lines.append(f"- `{name}`: {stats['mean']:+.6f} ± {stats['std']:.6f}")
        lines.append("")

    family_rows: list[str] = []
    for run_id, final in sorted(finals.items()):
        if final["dataset"] != "temporal_logic":
            continue
        per_family = final["metrics"]["test"].get("per_family", {})
        for family, metrics in sorted(per_family.items()):
            family_rows.append(
                f"| {final['variant']} | {final['seed']} | {family} | {metrics['accuracy']:.6f} |"
            )
    if family_rows:
        lines.extend(
            [
                "## Temporal formula families",
                "",
                "| Variant | Seed | Family | Accuracy |",
                "|---|---:|---|---:|",
                *family_rows,
                "",
            ]
        )

    order_rows: list[str] = []
    for final in finals.values():
        if final["variant"] not in {"time_shuffle", "time_reverse"}:
            continue
        test = final["metrics"]["test"]
        original = final["metrics"].get("original_test")
        frozen = test.get("frozen_label_metrics", {})
        order_rows.append(
            "| {dataset} | {variant} | {seed} | {valid:.6f} | {frozen:.6f} | {original} |".format(
                dataset=final["dataset"],
                variant=final["variant"],
                seed=final["seed"],
                valid=test["accuracy"],
                frozen=float(frozen.get("accuracy", float("nan"))),
                original=("—" if original is None else f"{original['accuracy']:.6f}"),
            )
        )
    if order_rows:
        lines.extend(
            [
                "## Time-order controls",
                "",
                "| Dataset | Variant | Seed | Valid-label accuracy | Frozen-label accuracy | Original-order accuracy |",
                "|---|---|---:|---:|---:|---:|",
                *order_rows,
                "",
            ]
        )

    lines.extend(
        [
            "## Provenance",
            "",
            f"- Git commit: `{summary['validation']['git_commit']}`",
            f"- Dataset manifests: `{json.dumps(summary['validation']['dataset_manifest_hashes'], sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_matrix(
    artifact_root: str | Path,
    specs: Sequence[RunSpec] | None = None,
    *,
    report_path: str | Path = "docs/experiment_report.md",
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    specs = expand_matrix() if specs is None else list(specs)
    validation = validate_matrix(artifact_root, specs)
    finals = validation.pop("finals")
    scores: dict[tuple[str, str], dict[int, float]] = {}
    for spec in specs:
        final = finals[spec.run_id]
        scores.setdefault((spec.dataset, spec.variant), {})[spec.seed] = float(
            final["metrics"]["test"]["accuracy"]
        )

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for (dataset, variant), seed_scores in scores.items():
        groups.setdefault(dataset, {})[variant] = _mean_std(
            seed_scores[seed] for seed in sorted(seed_scores)
        )
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in sorted(groups):
        for upper, lower in _PAIRS:
            if (dataset, upper) not in scores or (dataset, lower) not in scores:
                continue
            shared_seeds = sorted(set(scores[(dataset, upper)]) & set(scores[(dataset, lower)]))
            name = f"{upper}-{lower}"
            paired.setdefault(dataset, {})[name] = _mean_std(
                scores[(dataset, upper)][seed] - scores[(dataset, lower)][seed]
                for seed in shared_seeds
            )

    summary = {
        "schema_version": 1,
        "validation": validation,
        "groups": groups,
        "paired_deltas": paired,
    }
    _atomic_text(
        artifact_root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["dataset", "variant", "n", "test_accuracy_mean", "test_accuracy_sample_std"])
    for dataset in sorted(groups):
        for variant in VARIANTS:
            if variant not in groups[dataset]:
                continue
            stats = groups[dataset][variant]
            writer.writerow([dataset, variant, stats["n"], stats["mean"], stats["std"]])
    _atomic_text(artifact_root / "summary.csv", csv_buffer.getvalue())
    _atomic_text(Path(report_path), _report_markdown(summary, finals))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=Path("docs/experiment_report.md"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect-full", action="store_true")
    group.add_argument("--expect-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = (
        expand_matrix()
        if args.expect_full
        else expand_matrix(datasets=DATASETS, variants=VARIANTS, seeds=(42,))
    )
    summary = summarize_matrix(args.artifact_root, specs, report_path=args.report_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
