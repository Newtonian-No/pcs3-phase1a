import hashlib

import numpy as np
import pytest

from temporal_mamba.datasets import generalized_dynamics as dynamics
from temporal_mamba.datasets.generalized_dynamics import (
    DYNAMICS_SPLITS,
    GeneralizedDynamicsDataset,
    build_generalized_dynamics_manifest,
)


def _sizes(size: int) -> dict[str, int]:
    return {split: size for split in DYNAMICS_SPLITS}


def _content_fingerprint(item: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(item["coordinate_targets"][:, 0, :]).tobytes())
    digest.update(np.asarray([item["target"]], dtype="<i8").tobytes())
    return digest.hexdigest()


def test_manifest_is_reproducible_balanced_and_disjoint(tmp_path):
    sizes = _sizes(12)
    a = build_generalized_dynamics_manifest(tmp_path / "a", 20260717, sizes, signal_dim=3)
    b = build_generalized_dynamics_manifest(tmp_path / "b", 20260717, sizes, signal_dim=3)

    assert a["manifest_sha256"] == b["manifest_sha256"]
    assert a["files"] == b["files"]
    assert a["cross_split_duplicates"] == 0
    assert a["cross_split_sample_id_duplicates"] == 0
    datasets = {split: GeneralizedDynamicsDataset(tmp_path / "a", split) for split in sizes}
    ids = [{datasets[s][i]["sample_id"] for i in range(len(datasets[s]))} for s in sizes]
    assert sum(len(group) for group in ids) == len(set().union(*ids))

    fingerprints: set[str] = set()
    for split, dataset in datasets.items():
        labels = [dataset[i]["target"] for i in range(len(dataset))]
        assert np.bincount(labels).tolist() == [4, 4, 4]
        assert a["label_counts"][split] == {"0": 4, "1": 4, "2": 4}
        for index in range(len(dataset)):
            fingerprint = _content_fingerprint(dataset[index])
            assert fingerprint not in fingerprints
            fingerprints.add(fingerprint)


def test_analytic_coordinates_satisfy_signal_contract(tmp_path):
    build_generalized_dynamics_manifest(tmp_path, 20260717, _sizes(6), signal_dim=2)
    item = GeneralizedDynamicsDataset(tmp_path, "train")[0]

    assert item["signal"].shape == (128, 2)
    assert item["coordinate_targets"].shape == (128, 3, 2)
    assert item["coordinate_mask"].shape == (128, 3, 1)
    assert item["features"].shape == (128, 3)
    assert item["signal"].dtype == np.float32
    assert item["coordinate_targets"].dtype == np.float32
    assert item["features"].dtype == np.float32
    np.testing.assert_array_equal(item["coordinate_targets"][:, 0], item["signal"])
    np.testing.assert_array_equal(item["coordinate_mask"], np.ones((128, 3, 1), dtype=np.float32))
    assert item["target"] == item["base_target"]
    assert item["formula_family"] in {"damped", "forced", "switching"}


def test_all_formula_families_have_exact_first_and_second_derivatives():
    t = np.asarray([0.0, 0.2, 0.7, 1.1], dtype=np.float64)
    amp, phase = 1.3, 0.4

    damping, omega = 0.2, 2.1
    angle = omega * t + phase
    decay = np.exp(-damping * t)
    damped_x = amp * decay * np.cos(angle)
    damped_dx = amp * decay * (-damping * np.cos(angle) - omega * np.sin(angle))
    damped_ddx = -2 * damping * damped_dx - (damping**2 + omega**2) * damped_x
    actual = dynamics.damped(t, amp, phase, damping, omega)
    for value, expected in zip(actual, (damped_x, damped_dx, damped_ddx)):
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)

    drive_omega = 4.3
    forced_x = amp * np.cos(omega * t + phase) + 0.5 * amp * np.cos(drive_omega * t - phase)
    forced_dx = (
        -amp * omega * np.sin(omega * t + phase)
        - 0.5 * amp * drive_omega * np.sin(drive_omega * t - phase)
    )
    forced_ddx = (
        -amp * omega**2 * np.cos(omega * t + phase)
        - 0.5 * amp * drive_omega**2 * np.cos(drive_omega * t - phase)
    )
    actual = dynamics.forced(t, amp, phase, omega, drive_omega)
    for value, expected in zip(actual, (forced_x, forced_dx, forced_ddx)):
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)

    omega_before, omega_after, switch_index = 1.5, 3.2, 2
    switch_angle = omega_before * t[switch_index] + phase
    switching_angle = np.concatenate(
        [
            omega_before * t[:switch_index] + phase,
            switch_angle + omega_after * (t[switch_index:] - t[switch_index]),
        ]
    )
    switching_omega = np.asarray([omega_before, omega_before, omega_after, omega_after])
    switching_x = amp * np.cos(switching_angle)
    switching_dx = -amp * switching_omega * np.sin(switching_angle)
    switching_ddx = -amp * switching_omega**2 * np.cos(switching_angle)
    actual = dynamics.switching(t, amp, phase, omega_before, omega_after, switch_index)
    for value, expected in zip(actual, (switching_x, switching_dx, switching_ddx)):
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-13)


def test_coordinate_normalization_uses_train_signal_std_for_every_order():
    raw = np.asarray(
        [
            [
                [[12.0, 6.0], [4.0, 8.0], [-8.0, 12.0]],
                [[8.0, -6.0], [2.0, -4.0], [6.0, 20.0]],
            ]
        ],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            [
                [[1.0, 2.0], [2.0, 2.0], [-4.0, 3.0]],
                [[-1.0, -1.0], [1.0, -1.0], [3.0, 5.0]],
            ]
        ],
        dtype=np.float64,
    )

    normalized = dynamics._normalize_coordinates(
        raw,
        mean=np.asarray([10.0, -2.0]),
        std=np.asarray([2.0, 4.0]),
    )

    np.testing.assert_array_equal(normalized, expected)
    np.testing.assert_array_equal(raw[0, 0, 1], [4.0, 8.0])


def test_noise_cannot_hide_clean_trajectory_overlap(monkeypatch, tmp_path):
    original_generate = dynamics._generate_raw_split
    generated_test: dict[str, np.ndarray] = {}

    def generate_with_leak(split, size, signal_dim, seq_len, rng):
        if split == "noise_ood":
            return {name: value.copy() for name, value in generated_test.items()}
        generated = original_generate(split, size, signal_dim, seq_len, rng)
        if split == "test":
            generated_test.update({name: value.copy() for name, value in generated.items()})
        return generated

    monkeypatch.setattr(dynamics, "_generate_raw_split", generate_with_leak)

    with pytest.raises(RuntimeError, match="content fingerprints overlap"):
        build_generalized_dynamics_manifest(tmp_path, 20260717, _sizes(3), signal_dim=1)


def test_fingerprint_identity_is_clean_order_zero_trajectory_plus_label():
    coordinates_a = np.zeros((8, 3, 2), dtype=np.float32)
    coordinates_b = coordinates_a.copy()
    coordinates_b[:, 1:, :] = 17.0

    assert dynamics._content_fingerprint(coordinates_a, 2) == dynamics._content_fingerprint(coordinates_b, 2)
    assert dynamics._content_fingerprint(coordinates_a, 1) != dynamics._content_fingerprint(coordinates_b, 2)


def test_lengths_training_normalization_and_noise_only_observation(tmp_path):
    manifest = build_generalized_dynamics_manifest(tmp_path, 20260717, _sizes(6), signal_dim=2)
    train = GeneralizedDynamicsDataset(tmp_path, "train")
    length_256 = GeneralizedDynamicsDataset(tmp_path, "length_256")
    length_512 = GeneralizedDynamicsDataset(tmp_path, "length_512")
    noisy = GeneralizedDynamicsDataset(tmp_path, "noise_ood")

    stacked_train = np.stack([train[i]["signal"] for i in range(len(train))]).astype(np.float64)
    np.testing.assert_allclose(stacked_train.mean(axis=(0, 1)), 0.0, atol=2e-7)
    np.testing.assert_allclose(stacked_train.std(axis=(0, 1)), 1.0, atol=2e-7)
    assert length_256[0]["signal"].shape == (256, 2)
    assert length_512[0]["signal"].shape == (512, 2)
    assert not np.array_equal(noisy[0]["signal"], noisy[0]["coordinate_targets"][:, 0])
    assert manifest["normalization"]["source_split"] == "train"
    assert manifest["files"]["length_256"]["shape"] == [6, 256, 2]
    assert manifest["files"]["length_512"]["shape"] == [6, 512, 2]

    with np.load(tmp_path / "noise_ood.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["clean_signal"], data["coordinate_targets"][:, :, 0, :])
        assert np.any(data["signal"] != data["clean_signal"])


def test_manifest_records_strict_schema_ranges_and_hashes(tmp_path):
    manifest = build_generalized_dynamics_manifest(tmp_path, 20260717, _sizes(3), signal_dim=1)

    assert manifest["schema_version"] == 1
    assert manifest["generator_version"] == "generalized-dynamics-v1"
    assert set(manifest["ranges"]) == {"id", "parameter_ood", "noise_ood"}
    assert manifest["sizes"] == _sizes(3)
    assert manifest["shapes"]["coordinate_targets"] == [None, 3, 1]
    for split in DYNAMICS_SPLITS:
        path = tmp_path / manifest["files"][split]["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][split]["sha256"]

    with np.load(tmp_path / "parameter_ood.npz", allow_pickle=False) as data:
        assert np.all(data["parameter_regime"] == 1)
    with np.load(tmp_path / "train.npz", allow_pickle=False) as data:
        assert np.all(data["parameter_regime"] == 0)
        switching = data["family"] == 2
        assert np.all(data["switch_index"][switching] > 0)
        assert np.all(data["switch_rate"][switching] > 0.0)


@pytest.mark.parametrize(
    ("sizes", "data_seed", "signal_dim", "seq_len", "message"),
    [
        ({"train": 3}, 1, 1, 16, "exactly"),
        (_sizes(4), 1, 1, 16, "multiple of three"),
        (_sizes(3), 0, 1, 16, "data_seed"),
        (_sizes(3), 1, 0, 16, "signal_dim"),
        (_sizes(3), 1, 1, 1, "seq_len"),
    ],
)
def test_builder_rejects_invalid_contracts(tmp_path, sizes, data_seed, signal_dim, seq_len, message):
    with pytest.raises(ValueError, match=message):
        build_generalized_dynamics_manifest(
            tmp_path,
            data_seed,
            sizes,
            signal_dim=signal_dim,
            seq_len=seq_len,
        )


def test_dataset_rejects_unknown_split(tmp_path):
    build_generalized_dynamics_manifest(tmp_path, 20260717, _sizes(3), signal_dim=1)
    with pytest.raises(ValueError, match="split"):
        GeneralizedDynamicsDataset(tmp_path, "unknown")
