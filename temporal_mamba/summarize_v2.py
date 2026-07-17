"""Strict 12-run temporal_logic_v2 summary and attribution report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import TRAINING_SEEDS
from .datasets.temporal_logic import FORMULA_FAMILIES
from .run_matrix import V2_VARIANTS, RunSpec, expand_matrix
from .summarize import _atomic_text, validate_matrix
from .v2_gate import load_v2_final


V2_VIEWS = (
    "val",
    "test",
    "long_test",
    "channel_ood",
    "reverse_frozen",
    "shuffle_frozen",
)
_PAIRS = (
    ("two_pass", "vanilla"),
    ("error_inject", "two_pass"),
    ("error_aux", "error_inject"),
)


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


def _v2_specs() -> list[RunSpec]:
    return expand_matrix(
        datasets=("temporal_logic_v2",),
        variants=V2_VARIANTS,
        seeds=TRAINING_SEEDS,
    )


def _gate_status(root: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (root / "gate.json", root.parent / "v2-gate" / "gate.json")
    for path in candidates:
        if not path.exists():
            continue
        gate = json.loads(path.read_text(encoding="utf-8"))
        if gate.get("passed") is not True:
            raise ValueError(f"gate is not passed: {path}")
        if gate.get("git_commit") != validation["git_commit"]:
            raise ValueError("gate git_commit mismatch")
        expected_manifest = validation["dataset_manifest_hashes"]["temporal_logic_v2"]
        if gate.get("dataset_manifest_hash") != expected_manifest:
            raise ValueError("gate dataset_manifest_hash mismatch")
        return {"passed": True, "path": str(path)}
    return {"passed": None, "path": None, "status": "not_provided"}


def _raw_attribution(
    raw_root: Path | None,
    finals: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    if raw_root is None:
        return {"binder_minus_raw": None, "status": "raw_artifact_not_provided"}
    raw = load_v2_final(raw_root, variant="vanilla", seed=42)
    run_dir = raw_root / "temporal_logic_v2-vanilla-seed42"
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise ValueError("raw diagnostic missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("input_mode") != "raw_concat":
        raise ValueError("raw diagnostic input_mode must be raw_concat")
    expected_manifest = validation["dataset_manifest_hashes"]["temporal_logic_v2"]
    if raw["git_commit"] != validation["git_commit"]:
        raise ValueError("raw diagnostic git_commit mismatch")
    if raw["dataset_manifest_hash"] != expected_manifest:
        raise ValueError("raw diagnostic dataset_manifest_hash mismatch")
    bound = finals["temporal_logic_v2-vanilla-seed42"]
    bound_score = float(bound["metrics"]["test"]["balanced_accuracy"])
    raw_score = float(raw["metrics"]["test"]["balanced_accuracy"])
    return {
        "binder_minus_raw": bound_score - raw_score,
        "bound_test_balanced_accuracy": bound_score,
        "raw_test_balanced_accuracy": raw_score,
        "seed": 42,
        "status": "verified",
    }


def _report_markdown(summary: Mapping[str, Any]) -> str:
    validation = summary["validation"]
    gate = summary["gate"]
    lines = [
        "# Temporal Logic v2 experiment report",
        "",
        f"Verified runs: {validation['complete_runs']}/{validation['expected_runs']}",
        f"Stage-2 gate: {gate.get('passed')}",
        "",
        "## Evaluation views",
        "",
        "| View | Variant | Balanced accuracy mean | Sample std | Seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for view in V2_VIEWS:
        for variant in V2_VARIANTS:
            stats = summary["views"][view][variant]
            lines.append(
                f"| {view} | {variant} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['n']} |"
            )
    lines.extend(["", "## Formula families", "", "| Family | Variant | Test accuracy mean |", "|---|---|---:|"])
    for family in FORMULA_FAMILIES:
        for variant in V2_VARIANTS:
            stats = summary["families"][family][variant]
            lines.append(f"| {family} | {variant} | {stats['mean']:.6f} |")
    attribution = summary["attribution"]
    lines.extend(
        [
            "",
            "## Attribution and order controls",
            "",
            f"- Binder minus raw: `{attribution['binder_minus_raw']}`",
            f"- Reverse frozen minus ID: `{summary['order_controls']['reverse_minus_test']['mean']}`",
            f"- Shuffle frozen minus ID: `{summary['order_controls']['shuffle_minus_test']['mean']}`",
            "- Negative frozen-order deltas indicate sensitivity to temporal order under unchanged labels.",
            "",
            "## Provenance",
            "",
            f"- Git commit: `{validation['git_commit']}`",
            f"- Dataset manifest: `{validation['dataset_manifest_hashes']['temporal_logic_v2']}`",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_v2_matrix(
    artifact_root: str | Path,
    *,
    raw_artifact_root: str | Path | None = None,
    report_path: str | Path = "docs/temporal_logic_v2_report.md",
) -> dict[str, Any]:
    root = Path(artifact_root)
    specs = _v2_specs()
    expected_ids = {spec.run_id for spec in specs}
    discovered_ids = {
        path.parent.name for path in root.glob("temporal_logic_v2-*-seed*/final.json")
    }
    unexpected = sorted(discovered_ids - expected_ids)
    if unexpected:
        raise ValueError(f"unexpected v2 final artifacts: {unexpected}")
    validation = validate_matrix(root, specs)
    finals = validation.pop("finals")
    for spec in specs:
        final = finals[spec.run_id]
        if final.get("schema_version") != 2:
            raise ValueError(f"{spec.run_id} schema_version must be 2")
        if set(final["metrics"]) != set(V2_VIEWS):
            raise ValueError(f"{spec.run_id} metrics views mismatch")
        for view in V2_VIEWS:
            if "balanced_accuracy" not in final["metrics"][view]:
                raise ValueError(f"{spec.run_id} {view} missing balanced_accuracy")
        families = final["metrics"]["test"].get("per_family", {})
        if set(families) != set(FORMULA_FAMILIES):
            raise ValueError(f"{spec.run_id} test formula families mismatch")

    score_by_view: dict[str, dict[str, dict[int, float]]] = {
        view: {variant: {} for variant in V2_VARIANTS} for view in V2_VIEWS
    }
    family_scores: dict[str, dict[str, dict[int, float]]] = {
        family: {variant: {} for variant in V2_VARIANTS} for family in FORMULA_FAMILIES
    }
    for spec in specs:
        final = finals[spec.run_id]
        for view in V2_VIEWS:
            score_by_view[view][spec.variant][spec.seed] = float(
                final["metrics"][view]["balanced_accuracy"]
            )
        for family in FORMULA_FAMILIES:
            family_scores[family][spec.variant][spec.seed] = float(
                final["metrics"]["test"]["per_family"][family]["accuracy"]
            )

    views = {
        view: {
            variant: _mean_std(scores[seed] for seed in sorted(scores))
            for variant, scores in variants.items()
        }
        for view, variants in score_by_view.items()
    }
    families = {
        family: {
            variant: _mean_std(scores[seed] for seed in sorted(scores))
            for variant, scores in variants.items()
        }
        for family, variants in family_scores.items()
    }
    paired_deltas: dict[str, dict[str, dict[str, Any]]] = {}
    for view in V2_VIEWS:
        for upper, lower in _PAIRS:
            paired_deltas.setdefault(view, {})[f"{upper}-{lower}"] = _mean_std(
                score_by_view[view][upper][seed] - score_by_view[view][lower][seed]
                for seed in TRAINING_SEEDS
            )
    order_controls = {
        "reverse_minus_test": _mean_std(
            score_by_view["reverse_frozen"][variant][seed]
            - score_by_view["test"][variant][seed]
            for variant in V2_VARIANTS
            for seed in TRAINING_SEEDS
        ),
        "shuffle_minus_test": _mean_std(
            score_by_view["shuffle_frozen"][variant][seed]
            - score_by_view["test"][variant][seed]
            for variant in V2_VARIANTS
            for seed in TRAINING_SEEDS
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": 2,
        "validation": validation,
        "gate": _gate_status(root, validation),
        "views": views,
        "families": families,
        "paired_deltas": paired_deltas,
        "order_controls": order_controls,
        "attribution": _raw_attribution(
            None if raw_artifact_root is None else Path(raw_artifact_root),
            finals,
            validation,
        ),
    }
    _atomic_text(
        root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["view", "variant", "n", "balanced_accuracy_mean", "sample_std"])
    for view in V2_VIEWS:
        for variant in V2_VARIANTS:
            stats = views[view][variant]
            writer.writerow([view, variant, stats["n"], stats["mean"], stats["std"]])
    _atomic_text(root / "summary.csv", buffer.getvalue())
    _atomic_text(Path(report_path), _report_markdown(summary))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--raw-artifact-root", type=Path)
    parser.add_argument("--report-path", type=Path, default=Path("docs/temporal_logic_v2_report.md"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_v2_matrix(
        args.artifact_root,
        raw_artifact_root=args.raw_artifact_root,
        report_path=args.report_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

