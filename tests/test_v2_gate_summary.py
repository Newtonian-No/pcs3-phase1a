import json

import pytest

from temporal_mamba.datasets.temporal_logic import FORMULA_FAMILIES
from temporal_mamba.summarize_v2 import summarize_v2_matrix
from temporal_mamba.v2_gate import validate_v2_gate


V2_VARIANTS = ("vanilla", "two_pass", "error_inject", "error_aux")
V2_SEEDS = (42, 123, 777)
V2_VIEWS = (
    "val",
    "test",
    "long_test",
    "channel_ood",
    "reverse_frozen",
    "shuffle_frozen",
)


def _view(score: float, family_min: float) -> dict[str, object]:
    return {
        "accuracy": score,
        "balanced_accuracy": score,
        "per_family": {
            family: {"accuracy": family_min + 0.001 * index, "balanced_accuracy": family_min}
            for index, family in enumerate(FORMULA_FAMILIES)
        },
        "diagnostics": {
            "dt_min": 0.001,
            "dt_max": 0.1,
            "finite": True,
        },
    }


def write_v2_final(
    root,
    *,
    variant="vanilla",
    seed=42,
    overall=0.82,
    family_min=0.72,
    ood=0.75,
    commit="abc",
    manifest="data",
    input_mode=None,
):
    run_id = f"temporal_logic_v2-{variant}-seed{seed}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        view: _view(overall + 0.001 * index, family_min)
        for index, view in enumerate(V2_VIEWS)
    }
    metrics["channel_ood"] = _view(ood, family_min)
    final = {
        "schema_version": 2,
        "status": "complete",
        "run_id": run_id,
        "dataset": "temporal_logic_v2",
        "variant": variant,
        "seed": seed,
        "config_hash": f"config-{variant}-{seed}",
        "dataset_manifest_hash": manifest,
        "git_commit": commit,
        "metrics": metrics,
    }
    (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")
    if input_mode is not None:
        (run_dir / "config.json").write_text(
            json.dumps({"dataset": "temporal_logic_v2", "input_mode": input_mode}),
            encoding="utf-8",
        )
    return run_dir


def test_v2_gate_requires_every_seed_and_family(tmp_path):
    write_v2_final(tmp_path, seed=42, overall=0.82, family_min=0.72, ood=0.75)
    write_v2_final(tmp_path, seed=123, overall=0.83, family_min=0.71, ood=0.74)

    result = validate_v2_gate(tmp_path)

    assert result["passed"] is True
    assert set(result["seeds"]) == {"42", "123"}
    assert (tmp_path / "gate.json").exists()
    path = tmp_path / "temporal_logic_v2-vanilla-seed123" / "final.json"
    final = json.loads(path.read_text(encoding="utf-8"))
    final["metrics"]["val"]["per_family"]["GAP"]["accuracy"] = 0.69
    path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="GAP"):
        validate_v2_gate(tmp_path)


def test_v2_gate_rejects_missing_seed_and_mixed_provenance(tmp_path):
    write_v2_final(tmp_path, seed=42)
    with pytest.raises(ValueError, match="seed123"):
        validate_v2_gate(tmp_path)
    write_v2_final(tmp_path, seed=123, commit="different")
    with pytest.raises(ValueError, match="git_commit"):
        validate_v2_gate(tmp_path)


def _write_complete_matrix(root, *, commit="abc", manifest="data"):
    for variant_index, variant in enumerate(V2_VARIANTS):
        for seed in V2_SEEDS:
            write_v2_final(
                root,
                variant=variant,
                seed=seed,
                overall=0.80 + 0.01 * variant_index + seed / 1_000_000,
                family_min=0.72,
                ood=0.75,
                commit=commit,
                manifest=manifest,
            )


def test_v2_summary_requires_12_matching_finals_and_raw_attribution(tmp_path):
    full_root = tmp_path / "full"
    raw_root = tmp_path / "raw"
    _write_complete_matrix(full_root)
    write_v2_final(
        raw_root,
        seed=42,
        overall=0.55,
        family_min=0.50,
        ood=0.52,
        input_mode="raw_concat",
    )

    summary = summarize_v2_matrix(
        full_root,
        raw_artifact_root=raw_root,
        report_path=tmp_path / "report.md",
    )

    assert summary["validation"]["complete_runs"] == 12
    assert summary["attribution"]["binder_minus_raw"] > 0
    assert set(summary["views"]) == set(V2_VIEWS)
    assert (full_root / "summary.json").exists()
    assert (full_root / "summary.csv").exists()
    assert (tmp_path / "report.md").exists()

    missing = full_root / "temporal_logic_v2-error_aux-seed777" / "final.json"
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        summarize_v2_matrix(full_root, report_path=tmp_path / "report.md")


def test_v2_summary_rejects_extra_runs_and_mixed_commits(tmp_path):
    _write_complete_matrix(tmp_path)
    write_v2_final(tmp_path, variant="vanilla", seed=999)
    with pytest.raises(ValueError, match="unexpected"):
        summarize_v2_matrix(tmp_path, report_path=tmp_path / "report.md")

    extra = tmp_path / "temporal_logic_v2-vanilla-seed999"
    for path in extra.iterdir():
        path.unlink()
    extra.rmdir()
    path = tmp_path / "temporal_logic_v2-error_aux-seed777" / "final.json"
    final = json.loads(path.read_text(encoding="utf-8"))
    final["git_commit"] = "mixed"
    path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="git_commit"):
        summarize_v2_matrix(tmp_path, report_path=tmp_path / "report.md")
