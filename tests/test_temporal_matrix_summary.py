import json

import pytest

from temporal_mamba.config import TRAINING_SEEDS, VARIANTS
from temporal_mamba.run_matrix import expand_matrix, run_matrix
from temporal_mamba.summarize import summarize_matrix, validate_matrix


def test_full_matrix_is_exact_and_unique():
    specs = expand_matrix(datasets=("temporal_logic", "uci_har"))
    assert len(specs) == 36
    assert len({spec.run_id for spec in specs}) == 36
    assert {spec.seed for spec in specs} == set(TRAINING_SEEDS)
    assert {spec.variant for spec in specs} == set(VARIANTS)


def _write_fake_final(root, spec, score):
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True)
    final = {
        "schema_version": 1,
        "status": "complete",
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "seed": spec.seed,
        "config_hash": f"config-{spec.run_id}",
        "dataset_manifest_hash": f"data-{spec.dataset}",
        "git_commit": "commit-a",
        "pass_count": 1 if spec.variant == "vanilla" else 2,
        "uses_error": spec.variant not in {"vanilla", "two_pass"},
        "uses_aux": spec.variant in {"error_aux", "time_shuffle", "time_reverse"},
        "time_transform": {
            "time_shuffle": "shuffle",
            "time_reverse": "reverse",
        }.get(spec.variant, "none"),
        "metrics": {
            "test": {
                "accuracy": score,
                "diagnostics": {"dt_min": 0.001, "dt_max": 0.1, "finite": True},
                "per_family": {},
                "confusion_matrix": [[1, 0], [0, 1]],
            }
        },
    }
    (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")


def test_completeness_and_paired_deltas(tmp_path):
    specs = expand_matrix()
    variant_index = {variant: index for index, variant in enumerate(VARIANTS)}
    for spec in specs[:-1]:
        score = 0.5 + 0.01 * variant_index[spec.variant] + 0.00001 * spec.seed
        _write_fake_final(tmp_path, spec, score)
    with pytest.raises(ValueError, match=specs[-1].run_id):
        validate_matrix(tmp_path, specs)

    last = specs[-1]
    _write_fake_final(
        tmp_path,
        last,
        0.5 + 0.01 * variant_index[last.variant] + 0.00001 * last.seed,
    )
    validation = validate_matrix(tmp_path, specs)
    assert validation["complete_runs"] == 36
    report_path = tmp_path / "report.md"
    summary = summarize_matrix(tmp_path, specs, report_path=report_path)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert report_path.exists()
    for dataset in ("temporal_logic", "uci_har"):
        assert summary["paired_deltas"][dataset]["two_pass-vanilla"]["mean"] == pytest.approx(0.01)
        assert summary["paired_deltas"][dataset]["error_inject-two_pass"]["mean"] == pytest.approx(0.01)
        assert summary["paired_deltas"][dataset]["error_aux-error_inject"]["mean"] == pytest.approx(0.01)
        assert summary["paired_deltas"][dataset]["two_pass-vanilla"]["std"] == pytest.approx(0.0)


def test_validation_rejects_metadata_and_numerical_mismatch(tmp_path):
    spec = expand_matrix(datasets=("temporal_logic",), variants=("vanilla",), seeds=(42,))[0]
    _write_fake_final(tmp_path, spec, 0.7)
    path = tmp_path / spec.run_id / "final.json"
    final = json.loads(path.read_text(encoding="utf-8"))
    final["variant"] = "two_pass"
    path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="variant"):
        validate_matrix(tmp_path, [spec])
    final["variant"] = "vanilla"
    final["metrics"]["test"]["diagnostics"]["dt_max"] = 0.2
    path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="dt_max"):
        validate_matrix(tmp_path, [spec])


def test_matrix_dry_run_lists_without_mutation(tmp_path):
    specs = expand_matrix(
        datasets=("temporal_logic",),
        variants=("vanilla", "two_pass"),
        seeds=(42,),
    )
    result = run_matrix(
        specs,
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        dry_run=True,
    )
    assert result == [spec.run_id for spec in specs]
    assert not (tmp_path / "artifacts").exists()
