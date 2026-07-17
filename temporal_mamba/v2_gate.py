"""Strict two-seed generalization gate for temporal_logic_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .datasets.temporal_logic import FORMULA_FAMILIES
from .summarize import _atomic_text, _check_numeric_tree


def load_v2_final(
    artifact_root: str | Path,
    *,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    run_id = f"temporal_logic_v2-{variant}-seed{seed}"
    path = Path(artifact_root) / run_id / "final.json"
    if not path.exists():
        raise ValueError(f"missing final artifact: {run_id}")
    try:
        final = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid final artifact JSON: {run_id}: {exc}") from exc
    for field, expected in (
        ("schema_version", 2),
        ("status", "complete"),
        ("run_id", run_id),
        ("dataset", "temporal_logic_v2"),
        ("variant", variant),
        ("seed", seed),
    ):
        if final.get(field) != expected:
            raise ValueError(
                f"{run_id} {field} mismatch: expected {expected!r}, got {final.get(field)!r}"
            )
    for field in ("config_hash", "dataset_manifest_hash", "git_commit", "metrics"):
        if field not in final or final[field] in (None, ""):
            raise ValueError(f"{run_id} missing metadata: {field}")
    _check_numeric_tree(final["metrics"], f"{run_id}.metrics")
    return final


def _require_shared_provenance(finals: Iterable[dict[str, Any]]) -> tuple[str, str]:
    finals = list(finals)
    commits = {str(final["git_commit"]) for final in finals}
    manifests = {str(final["dataset_manifest_hash"]) for final in finals}
    if len(commits) != 1:
        raise ValueError(f"git_commit mismatch across v2 artifacts: {sorted(commits)}")
    if len(manifests) != 1:
        raise ValueError(
            f"dataset_manifest_hash mismatch across v2 artifacts: {sorted(manifests)}"
        )
    return next(iter(commits)), next(iter(manifests))


def validate_v2_gate(
    artifact_root: str | Path,
    seeds: tuple[int, ...] = (42, 123),
    overall: float = 0.80,
    family: float = 0.70,
    ood: float = 0.70,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("gate requires at least one seed")
    if any(not 0 <= threshold <= 1 for threshold in (overall, family, ood)):
        raise ValueError("gate thresholds must be in [0, 1]")
    root = Path(artifact_root)
    finals = [load_v2_final(root, variant="vanilla", seed=seed) for seed in seeds]
    commit, manifest = _require_shared_provenance(finals)
    seed_results: dict[str, dict[str, Any]] = {}
    for seed, final in zip(seeds, finals):
        metrics = final["metrics"]
        for view in ("val", "channel_ood"):
            if view not in metrics:
                raise ValueError(f"seed{seed} missing metrics view: {view}")
        val_score = float(metrics["val"].get("balanced_accuracy", -1.0))
        if val_score < overall:
            raise ValueError(
                f"seed{seed} val balanced_accuracy {val_score:.6f} below {overall:.6f}"
            )
        per_family = metrics["val"].get("per_family", {})
        if set(per_family) != set(FORMULA_FAMILIES):
            missing = sorted(set(FORMULA_FAMILIES) - set(per_family))
            extra = sorted(set(per_family) - set(FORMULA_FAMILIES))
            raise ValueError(f"seed{seed} formula families mismatch: missing={missing}, extra={extra}")
        family_scores: dict[str, float] = {}
        for family_name in FORMULA_FAMILIES:
            score = float(per_family[family_name].get("accuracy", -1.0))
            if score < family:
                raise ValueError(
                    f"seed{seed} {family_name} accuracy {score:.6f} below {family:.6f}"
                )
            family_scores[family_name] = score
        ood_score = float(metrics["channel_ood"].get("balanced_accuracy", -1.0))
        if ood_score < ood:
            raise ValueError(
                f"seed{seed} channel_ood balanced_accuracy {ood_score:.6f} below {ood:.6f}"
            )
        seed_results[str(seed)] = {
            "val_balanced_accuracy": val_score,
            "family_accuracy": family_scores,
            "channel_ood_balanced_accuracy": ood_score,
            "config_hash": final["config_hash"],
        }

    result: dict[str, Any] = {
        "schema_version": 1,
        "passed": True,
        "thresholds": {"overall": overall, "family": family, "ood": ood},
        "seeds": seed_results,
        "git_commit": commit,
        "dataset_manifest_hash": manifest,
    }
    _atomic_text(
        root / "gate.json",
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--overall", type=float, default=0.80)
    parser.add_argument("--family", type=float, default=0.70)
    parser.add_argument("--ood", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_v2_gate(
        args.artifact_root,
        seeds=tuple(args.seeds),
        overall=args.overall,
        family=args.family,
        ood=args.ood,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
