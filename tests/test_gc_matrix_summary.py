import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from temporal_mamba import run_gc_matrix as gc_runner_module
from temporal_mamba import summarize_gc as gc_summary_module
from temporal_mamba.config import GC_MATRIX_VARIANTS, load_experiment_config
from temporal_mamba.run_gc_matrix import (
    GC_CONFIG_NAMES,
    expand_gc_matrix,
    run_gc_matrix,
)
from temporal_mamba.summarize_gc import summarize_gc_matrix
from temporal_mamba.train import _config_payload


def _canonical_hash(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _effective_config(spec, stage="confirm"):
    config = load_experiment_config(
        gc_runner_module.Path("configs") / GC_CONFIG_NAMES[spec.dataset],
        variant=spec.variant,
        seed=spec.seed,
    )
    if stage == "smoke":
        config = replace(config, training=replace(config.training, epochs=1))
    return _config_payload(config)


def _manifest(dataset):
    if dataset == "generalized_dynamics":
        payload = {
            "schema_version": 1,
            "generator_version": "generalized-dynamics-v1",
            "data_seed": 20260717,
            "signal_dim": 6,
            "seq_len": 128,
            "formula_families": ["damped", "forced", "switching"],
            "splits": [
                "train",
                "val",
                "test",
                "length_256",
                "length_512",
                "parameter_ood",
                "noise_ood",
            ],
            "sizes": {
                "train": 18000,
                "val": 3000,
                "test": 3000,
                "length_256": 3000,
                "length_512": 3000,
                "parameter_ood": 3000,
                "noise_ood": 3000,
            },
            "shapes": {
                "signal": [None, 6],
                "coordinate_targets": [None, 3, 6],
                "coordinate_mask": [None, 3, 1],
                "features": [None, 7],
            },
            "ranges": {"id": {}},
            "normalization": {"source_split": "train", "mean": [0.0], "std": [1.0]},
            "files": {
                name: {"name": f"{name}.npz", "sha256": name}
                for name in (
                    "train",
                    "val",
                    "test",
                    "length_256",
                    "length_512",
                    "parameter_ood",
                    "noise_ood",
                )
            },
            "label_counts": {},
            "cross_split_duplicates": 0,
            "cross_split_sample_id_duplicates": 0,
        }
    else:
        payload = {
            "schema_version": 1,
            "data_seed": 20260716,
            "source_manifest_sha256": "source-sha256",
            "signal_names": [f"signal_{index}" for index in range(9)],
            "official_shapes": {"train": [1, 128, 9], "test": [1, 128, 9]},
            "validation_subjects": [1],
            "subjects": {"train": [2], "val": [1], "test": [3]},
            "normalization_mean": [0.0] * 9,
            "normalization_std": [1.0] * 9,
            "files": {name: {"name": f"{name}.npz", "sha256": name} for name in ("train", "val", "test")},
        }
    return {**payload, "manifest_sha256": _canonical_hash(payload)}


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


def test_gc_runner_cli_requires_stage_and_dry_run_expands_only_gc_jobs(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["run_gc_matrix"])
    with pytest.raises(SystemExit):
        gc_runner_module._parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gc_matrix",
            "--stage",
            "smoke",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--data-root",
            str(tmp_path / "data"),
            "--dry-run",
        ],
    )
    gc_runner_module.main()

    assert len(capsys.readouterr().out.strip().splitlines()) == 14


def test_gc_summary_cli_requires_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_gc", "--artifact-root", str(tmp_path)],
    )
    with pytest.raises(SystemExit):
        gc_summary_module._parse_args()


def test_gc_summary_main_success_path(tmp_path, monkeypatch, capsys):
    artifact_root = tmp_path / "confirm"
    _write_confirm_matrix(
        artifact_root,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_gc",
            "--stage",
            "confirm",
            "--artifact-root",
            str(artifact_root),
        ],
    )

    gc_summary_module.main()

    output = json.loads(capsys.readouterr().out)
    assert output["completed_jobs"] == 70
    assert output["decision"] == "supported"


def test_gc_matrix_uses_only_gc_configs_and_variants():
    specs = expand_gc_matrix("confirm")

    assert {spec.dataset for spec in specs} == {"generalized_dynamics", "uci_har"}
    assert {spec.variant for spec in specs} == set(GC_MATRIX_VARIANTS)
    assert len({spec.run_id for spec in specs}) == len(specs)


def test_smoke_runs_tiny_gate_then_one_epoch_training(tmp_path, monkeypatch):
    spec = expand_gc_matrix("smoke")[0]
    calls = []
    config_dir = tmp_path / "configs"
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
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
    manifest = _manifest(spec.dataset)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(gc_runner_module, "_git_commit", lambda: "abc")
    expected = gc_runner_module._expected_metadata(
        spec,
        stage="smoke",
        config_dir=config_dir,
        data_root=data_root,
    )

    def fake_run(command, *, check, env):
        calls.append(command)
        if "--overfit-only" not in command:
            run_dir = artifact_root / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "final.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "complete",
                        "run_id": spec.run_id,
                        "dataset": spec.dataset,
                        "variant": spec.variant,
                        "seed": spec.seed,
                        **expected,
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gc_runner_module.subprocess, "run", fake_run)
    completed = run_gc_matrix(
        "smoke",
        specs=(spec,),
        artifact_root=artifact_root,
        data_root=data_root,
        config_dir=config_dir,
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


def test_gc_runner_rejects_zero_exit_without_matching_final(tmp_path, monkeypatch):
    spec = expand_gc_matrix("screen")[0]
    config_dir = tmp_path / "configs"
    data_root = tmp_path / "data"
    config_dir.mkdir()
    source = gc_runner_module.Path("configs") / "generalized_dynamics_gc.json"
    (config_dir / "generalized_dynamics_gc.json").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest_dir = data_root / spec.dataset
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(_manifest(spec.dataset)), encoding="utf-8"
    )
    monkeypatch.setattr(gc_runner_module, "_git_commit", lambda: "abc")
    monkeypatch.setattr(
        gc_runner_module.subprocess,
        "run",
        lambda command, *, check, env: SimpleNamespace(returncode=0),
    )

    with pytest.raises(ValueError, match="completed final"):
        run_gc_matrix(
            "screen",
            specs=(spec,),
            artifact_root=tmp_path / "artifacts",
            data_root=data_root,
            config_dir=config_dir,
        )


@pytest.mark.parametrize(
    "field",
    ("run_id", "git_commit", "config_hash", "dataset_manifest_hash"),
)
def test_completed_artifact_reuse_requires_all_identity_hashes(
    tmp_path, monkeypatch, field
):
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
        json.dumps(_manifest(spec.dataset)), encoding="utf-8"
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

    final[field] = "wrong"
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
    manifest = manifest or _manifest(spec.dataset)
    manifest_hash = manifest["manifest_sha256"]
    config = _effective_config(spec)
    config_hash = _canonical_hash(config)
    gc_order = {
        "gc_k1": 1,
        "gc_k2": 2,
        "gc_k3": 3,
        "gc_k3_shuffled": 3,
        "gc_k3_noise": 3,
    }.get(spec.variant, 0)
    order_values = {
        f"order_{order}": 0.0 if order < gc_order else None for order in range(3)
    }
    final = {
        "schema_version": 3,
        "status": "complete",
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "variant": spec.variant,
        "seed": spec.seed,
        "config_hash": config_hash,
        "dataset_manifest_hash": manifest_hash,
        "git_commit": commit,
        "selection_split": "val",
        "best_epoch": 0,
        "best_metric": 0.70,
        "completed_epochs": config["training"]["epochs"],
        "global_step": 1,
        "pass_count": 1 if spec.variant == "vanilla" else 2,
        "uses_error": spec.variant not in {"vanilla", "two_pass"},
        "uses_aux": spec.variant not in {"vanilla", "two_pass"},
        "time_transform": "none",
        "gc_order": gc_order,
        "parameter_count": 1000 if spec.dataset == "generalized_dynamics" else 2000,
        "per_order_auxiliary_losses": order_values,
        "per_order_error_rms": order_values,
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
            "manifest_sha256": manifest_hash,
        },
        "environment": {
            "python": "3.10",
            "platform": "test",
            "torch": "test",
            "numpy": "test",
            "cuda_available": True,
            "cuda_version": "test",
            "device": "cuda",
            "device_name": "test",
            "git_commit": commit,
            "amp": False,
            "cpu_affinity": [0],
        },
        "metrics": metrics,
    }
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final.json").write_text(json.dumps(final), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


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
    persisted = json.loads((supported / "summary.json").read_text(encoding="utf-8"))
    assert persisted["completed_jobs"] == 70
    assert persisted["git_commit"] == "commit-a"
    assert persisted["dataset_manifest_hashes"] == {
        dataset: _manifest(dataset)["manifest_sha256"]
        for dataset in ("generalized_dynamics", "uci_har")
    }
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
    assert "generalized_dynamics config hashes" in report
    assert "uci_har config hashes" in report


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


@pytest.mark.parametrize(
    ("sidecar", "message"),
    (
        ("config.json", "config.json"),
        ("dataset_manifest.json", "dataset_manifest.json"),
    ),
)
def test_gc_summary_requires_every_run_sidecar(tmp_path, sidecar, message):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    spec = expand_gc_matrix("confirm")[0]
    (tmp_path / spec.run_id / sidecar).unlink()

    with pytest.raises(ValueError, match=message):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_recomputes_config_sidecar_hash(tmp_path):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    spec = expand_gc_matrix("confirm")[0]
    path = tmp_path / spec.run_id / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["seed"] = 999
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="config.*hash"):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_rejects_uniform_self_consistent_config_tampering(tmp_path):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    for spec in expand_gc_matrix("confirm"):
        run_dir = tmp_path / spec.run_id
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["seed"] = spec.seed + 1
        config["training"]["lr"] = 0.25
        config_hash = _canonical_hash(config)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        final_path = run_dir / "final.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["config_hash"] = config_hash
        final["hashes"]["config_sha256"] = config_hash
        final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="preregistered config"):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_rejects_uniformly_forged_manifest_claims(tmp_path):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    for spec in expand_gc_matrix("confirm"):
        run_dir = tmp_path / spec.run_id
        manifest_path = run_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["manifest_sha256"] = "forged"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        final_path = run_dir / "final.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["dataset_manifest_hash"] = "forged"
        final["hashes"]["manifest_sha256"] = "forged"
        final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.*canonical"):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_rejects_uniform_self_consistent_manifest_tampering(tmp_path):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    for spec in expand_gc_matrix("confirm"):
        run_dir = tmp_path / spec.run_id
        manifest_path = run_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["data_seed"] += 1
        payload = dict(manifest)
        payload.pop("manifest_sha256")
        manifest_hash = _canonical_hash(payload)
        manifest["manifest_sha256"] = manifest_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        final_path = run_dir / "final.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["dataset_manifest_hash"] = manifest_hash
        final["hashes"]["manifest_sha256"] = manifest_hash
        final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="data_seed"):
        summarize_gc_matrix(tmp_path, "confirm")


def test_gc_summary_rejects_uniform_self_consistent_dynamics_contract_tampering(
    tmp_path,
):
    _write_confirm_matrix(
        tmp_path,
        synthetic_effects=(0.030, 0.026, 0.034, 0.028, 0.032),
        uci_effects=(0.015, 0.013, 0.017, 0.014, 0.016),
    )
    for spec in expand_gc_matrix("confirm"):
        if spec.dataset != "generalized_dynamics":
            continue
        run_dir = tmp_path / spec.run_id
        manifest_path = run_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["signal_dim"] = 7
        manifest["seq_len"] = 256
        manifest["sizes"] = {name: value + 1 for name, value in manifest["sizes"].items()}
        manifest["shapes"]["signal"][-1] = 7
        manifest["shapes"]["coordinate_targets"][-1] = 7
        manifest["shapes"]["features"][-1] = 8
        manifest["formula_families"] = ["damped", "forced", "switching", "forged"]
        payload = dict(manifest)
        payload.pop("manifest_sha256")
        manifest_hash = _canonical_hash(payload)
        manifest["manifest_sha256"] = manifest_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        final_path = run_dir / "final.json"
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final["dataset_manifest_hash"] = manifest_hash
        final["hashes"]["manifest_sha256"] = manifest_hash
        final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="signal_dim|sizes"):
        summarize_gc_matrix(tmp_path, "confirm")


@pytest.mark.parametrize("invalid_size", (True, 1.5, 0, -1))
def test_gc_summary_rejects_nonpositive_or_noninteger_dynamics_sizes(
    tmp_path, invalid_size
):
    spec = expand_gc_matrix("confirm")[0]
    _write_gc_final(tmp_path, spec, scores={"id": 0.70, "ood": 0.70})
    run_dir = tmp_path / spec.run_id
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sizes"]["train"] = invalid_size
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    manifest_hash = _canonical_hash(payload)
    manifest["manifest_sha256"] = manifest_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    final_path = run_dir / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["dataset_manifest_hash"] = manifest_hash
    final["hashes"]["manifest_sha256"] = manifest_hash
    final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match="sizes"):
        gc_summary_module._validate_sidecars(
            tmp_path,
            spec,
            final,
            "confirm",
        )


@pytest.mark.parametrize(
    ("dataset", "field"),
    (
        ("generalized_dynamics", "generator_version"),
        ("uci_har", "source_manifest_sha256"),
    ),
)
def test_gc_summary_requires_dataset_specific_manifest_structure(
    tmp_path, dataset, field
):
    spec = next(spec for spec in expand_gc_matrix("confirm") if spec.dataset == dataset)
    _write_gc_final(tmp_path, spec, scores={"id": 0.70, "ood": 0.70})
    run_dir = tmp_path / spec.run_id
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(field)
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    manifest_hash = _canonical_hash(payload)
    manifest["manifest_sha256"] = manifest_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    final_path = run_dir / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["dataset_manifest_hash"] = manifest_hash
    final["hashes"]["manifest_sha256"] = manifest_hash
    final_path.write_text(json.dumps(final), encoding="utf-8")

    with pytest.raises(ValueError, match=field.replace("source_manifest_sha256", "source")):
        gc_summary_module._validate_sidecars(
            tmp_path,
            spec,
            final,
            "confirm",
        )
