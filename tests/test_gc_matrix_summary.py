import json
from types import SimpleNamespace

import pytest

from temporal_mamba import run_gc_matrix as gc_runner_module
from temporal_mamba.config import GC_MATRIX_VARIANTS
from temporal_mamba.run_gc_matrix import expand_gc_matrix, run_gc_matrix
from temporal_mamba.summarize_gc import summarize_gc_matrix


def test_gc_stage_matrix_sizes_and_preregistered_seeds():
    smoke = expand_gc_matrix("smoke")
    screen = expand_gc_matrix("screen")
    confirm = expand_gc_matrix("confirm")

    assert len(smoke) == 2 * 7
    assert len(screen) == 2 * 7 * 3
    assert len(confirm) == 2 * 7 * 5
    assert {run.seed for run in smoke} == {42}
    assert {run.seed for run in screen} == {42, 123, 777}
    assert {run.seed for run in confirm} == {42, 123, 777, 2026, 31415}


def test_gc_matrix_rejects_unknown_or_missing_stage():
    with pytest.raises(ValueError, match="stage"):
        expand_gc_matrix("legacy")
    with pytest.raises(TypeError):
        expand_gc_matrix()


def test_gc_matrix_uses_only_gc_configs_and_variants():
    specs = expand_gc_matrix("confirm")

    assert {spec.dataset for spec in specs} == {"generalized_dynamics", "uci_har"}
    assert {spec.variant for spec in specs} == set(GC_MATRIX_VARIANTS)
    assert len({spec.run_id for spec in specs}) == len(specs)


def test_smoke_runs_tiny_gate_then_one_epoch_training(tmp_path, monkeypatch):
    spec = expand_gc_matrix("smoke")[0]
    calls = []

    def fake_run(command, *, check, env):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gc_runner_module.subprocess, "run", fake_run)
    monkeypatch.setattr(gc_runner_module, "_git_commit", lambda: "abc")
    completed = run_gc_matrix(
        "smoke",
        specs=(spec,),
        artifact_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        config_dir=tmp_path / "configs",
    )

    assert completed == [spec.run_id]
    assert len(calls) == 2
    assert "--overfit-only" in calls[0]
    assert calls[0][calls[0].index("--epochs") + 1] == "1"
    assert "--overfit-only" not in calls[1]
    assert calls[1][calls[1].index("--epochs") + 1] == "1"
    assert calls[1][calls[1].index("--config") + 1].endswith(
        "generalized_dynamics_gc.json"
    )


def test_completed_artifact_reuse_requires_all_identity_hashes(tmp_path, monkeypatch):
    spec = expand_gc_matrix("confirm")[0]
    config_dir = tmp_path / "configs"
    data_root = tmp_path / "data"
    config_dir.mkdir()
    source_config = json.loads(
        (gc_runner_module.Path("configs") / "generalized_dynamics_gc.json").read_text(
            encoding="utf-8"
        )
    )
    (config_dir / "generalized_dynamics_gc.json").write_text(
        json.dumps(source_config), encoding="utf-8"
    )
    manifest_dir = data_root / spec.dataset
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"manifest_sha256": "manifest-a"}), encoding="utf-8"
    )
    monkeypatch.setattr(gc_runner_module, "_git_commit", lambda: "commit-a")
    expected = gc_runner_module._expected_metadata(
        spec,
        stage="confirm",
        config_dir=config_dir,
        data_root=data_root,
    )
    run_dir = tmp_path / "artifacts" / spec.run_id
    run_dir.mkdir(parents=True)
    final = {
        "schema_version": 3,
        "status": "complete",
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "seed": spec.seed,
        **expected,
    }
    path = run_dir / "final.json"
    path.write_text(json.dumps(final), encoding="utf-8")

    assert run_gc_matrix(
        "confirm",
        specs=(spec,),
        artifact_root=tmp_path / "artifacts",
        data_root=data_root,
        config_dir=config_dir,
    ) == [spec.run_id]

    final["dataset_manifest_hash"] = "wrong"
    path.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        run_gc_matrix(
            "confirm",
            specs=(spec,),
            artifact_root=tmp_path / "artifacts",
            data_root=data_root,
            config_dir=config_dir,
        )


def _metric(score, *, dataset):
    metric = {
        "accuracy": score,
        "balanced_accuracy": score,
        "diagnostics": {
            "dt_min": 0.001,
            "dt_max": 0.1,
            "finite": True,
        },
    }
    if dataset == "uci_har":
        metric["macro_f1"] = score
    return metric


def _write_gc_final(root, spec, *, scores, commit="commit-a", manifest=None):
    if spec.dataset == "generalized_dynamics":
        views = ("val", "test", "length_256", "length_512", "parameter_ood", "noise_ood")
    else:
        views = ("val", "test", "prefix50", "noise_025")
    metrics = {
        view: _metric(scores.get(view, scores["ood"] if view != "test" else scores["id"]), dataset=spec.dataset)
        for view in views
    }
    manifest = manifest or f"manifest-{spec.dataset}"
    config_hash = f"config-{spec.run_id}"
    final = {
        "schema_version": 3,
        "status": "complete",
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "seed": spec.seed,
        "config_hash": config_hash,
        "dataset_manifest_hash": manifest,
        "git_commit": commit,
        "gc_order": {
            "gc_k1": 1,
            "gc_k2": 2,
            "gc_k3": 3,
            "gc_k3_shuffled": 3,
            "gc_k3_noise": 3,
        }.get(spec.variant, 0),
        "parameter_count": 1000 if spec.dataset == "generalized_dynamics" else 2000,
        "per_order_auxiliary_losses": {"order_0": 0.0, "order_1": 0.0, "order_2": 0.0},
        "per_order_error_rms": {"order_0": 0.0, "order_1": 0.0, "order_2": 0.0},
        "dt_diagnostics": {
            view: {
                "dt_min": metrics[view]["diagnostics"]["dt_min"],
                "dt_max": metrics[view]["diagnostics"]["dt_max"],
            }
            for view in views
        },
        "hashes": {
            "git_commit": commit,
            "config_sha256": config_hash,
            "manifest_sha256": manifest,
        },
        "environment": {"git_commit": commit, "cuda_available": True},
        "metrics": metrics,
    }
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")


def _write_confirm_matrix(
    root, *, synthetic_effects, uci_effects, control_tie=False, id_drop=0.005
):
    effects = {
        "generalized_dynamics": dict(zip((42, 123, 777, 2026, 31415), synthetic_effects)),
        "uci_har": dict(zip((42, 123, 777, 2026, 31415), uci_effects)),
    }
    for spec in expand_gc_matrix("confirm"):
        baseline = 0.70
        effect = effects[spec.dataset][spec.seed]
        if spec.variant == "gc_k3":
            ood = baseline + effect
            id_score = baseline - id_drop
        elif spec.variant in {"gc_k3_shuffled", "gc_k3_noise"}:
            ood = baseline + effect if control_tie else baseline + effect - 0.01
            id_score = baseline
        else:
            ood = baseline
            id_score = baseline
        _write_gc_final(root, spec, scores={"id": id_score, "ood": ood})


def test_confirm_decision_has_supported_uncertain_and_not_supported(tmp_path):
    supported = tmp_path / "supported"
    one_domain = tmp_path / "one-domain"
    ci_contains_zero = tmp_path / "ci-contains-zero"
    below_threshold = tmp_path / "below-threshold"
    control_tie = tmp_path / "control-tie"
    _write_confirm_matrix(
        supported,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
        id_drop=0.01,
    )
    _write_confirm_matrix(
        one_domain,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.005, 0.004, 0.006, 0.005, 0.005),
    )
    _write_confirm_matrix(
        ci_contains_zero,
        synthetic_effects=(-0.020, 0.060, 0.020, 0.020, 0.020),
        uci_effects=(-0.010, 0.030, 0.010, 0.010, 0.010),
    )
    _write_confirm_matrix(
        below_threshold,
        synthetic_effects=(0.005, 0.004, 0.006, 0.005, 0.005),
        uci_effects=(0.005, 0.004, 0.006, 0.005, 0.005),
    )
    _write_confirm_matrix(
        control_tie,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
        control_tie=True,
    )

    supported_summary = summarize_gc_matrix(supported, "confirm")
    assert supported_summary["decision"] == "supported"
    assert supported_summary["domains"]["generalized_dynamics"]["k3_minus_two_pass"][
        "ci95"
    ]["critical_value"] == pytest.approx(2.776445105)
    assert summarize_gc_matrix(one_domain, "confirm")["decision"] == "uncertain"
    assert summarize_gc_matrix(ci_contains_zero, "confirm")["decision"] == "uncertain"
    assert summarize_gc_matrix(below_threshold, "confirm")["decision"] == "not_supported"
    assert summarize_gc_matrix(control_tie, "confirm")["decision"] == "not_supported"
    assert (supported / "summary.json").exists()
    report = (supported / "report_zh.md").read_text(encoding="utf-8")
    assert "结论：supported" in report
    assert "逐种子" in report


def test_screen_gate_requires_positive_mean_and_two_simultaneous_control_wins(tmp_path):
    for spec in expand_gc_matrix("screen"):
        if spec.variant == "gc_k3":
            score = {42: 0.73, 123: 0.72, 777: 0.69}[spec.seed]
        elif spec.variant in {"gc_k3_shuffled", "gc_k3_noise"}:
            score = 0.70
        else:
            score = 0.70
        _write_gc_final(tmp_path, spec, scores={"id": 0.70, "ood": score})

    summary = summarize_gc_matrix(tmp_path, "screen")

    assert summary["screen_gate"]["at_least_one_domain_passed"] is True
    assert summary["screen_gate"]["domains"]["generalized_dynamics"]["passed"] is True
    assert summary["decision"] == "uncertain"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("mixed_commit", "git_commit"),
        ("nonfinite", "non-finite"),
        ("invalid_dt", "dt_max"),
        ("wrong_manifest", "manifest"),
    ),
)
def test_gc_summary_rejects_invalid_artifacts(tmp_path, mutation, message):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    spec = expand_gc_matrix("confirm")[-1]
    path = tmp_path / spec.run_id / "final.json"
    final = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "mixed_commit":
        final["git_commit"] = "other"
        final["hashes"]["git_commit"] = "other"
        final["environment"]["git_commit"] = "other"
    elif mutation == "nonfinite":
        final["metrics"]["test"]["macro_f1"] = float("nan")
    elif mutation == "invalid_dt":
        final["metrics"]["test"]["diagnostics"]["dt_max"] = 0.2
    else:
        final["dataset_manifest_hash"] = "wrong"
        final["hashes"]["manifest_sha256"] = "wrong"
    path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_rejects_missing_job(tmp_path):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    missing = expand_gc_matrix("confirm")[-1]
    (tmp_path / missing.run_id / "final.json").unlink()

    with pytest.raises(ValueError, match="missing"):
        summarize_gc_matrix(tmp_path, "confirm")
