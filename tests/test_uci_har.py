import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from temporal_mamba.datasets import uci_har
from temporal_mamba.datasets.uci_har import (
    SIGNAL_NAMES,
    UCI_HAR_URL,
    UCIHARDataset,
    _safe_extract_zip,
    download_uci_har,
    prepare_uci_har,
)


def _write_partition(dataset_root: Path, partition: str, subjects: list[int]) -> None:
    inertial = dataset_root / partition / "Inertial Signals"
    inertial.mkdir(parents=True, exist_ok=True)
    subject_rows = np.repeat(np.asarray(subjects, dtype=np.int64), 2)
    count = len(subject_rows)
    time = np.linspace(-0.5, 0.5, 128, dtype=np.float64)
    for channel, name in enumerate(SIGNAL_NAMES):
        matrix = np.stack(
            [time + channel * 0.75 + row * 0.2 + subject_rows[row] * 0.01 for row in range(count)]
        )
        np.savetxt(inertial / f"{name}_{partition}.txt", matrix, fmt="%.8f")
    labels = np.asarray([(row % 6) + 1 for row in range(count)], dtype=np.int64)
    np.savetxt(dataset_root / partition / f"y_{partition}.txt", labels, fmt="%d")
    np.savetxt(dataset_root / partition / f"subject_{partition}.txt", subject_rows, fmt="%d")


def make_extracted_fixture(root: Path) -> Path:
    dataset_root = root / "extracted" / "UCI HAR Dataset"
    _write_partition(dataset_root, "train", list(range(1, 11)))
    _write_partition(dataset_root, "test", [11, 12])
    return dataset_root


def test_prepare_stacks_raw_signals_and_uses_subject_split(tmp_path):
    root = tmp_path / "har"
    make_extracted_fixture(root)
    manifest = prepare_uci_har(root, data_seed=20260716)
    train = UCIHARDataset(root, "train")
    val = UCIHARDataset(root, "val")
    test = UCIHARDataset(root, "test")

    expected_val = sorted(
        np.random.default_rng(20260716).choice(np.arange(1, 11), size=2, replace=False).tolist()
    )
    assert manifest["validation_subjects"] == expected_val
    train_subjects = {int(train[i]["subject_id"]) for i in range(len(train))}
    val_subjects = {int(val[i]["subject_id"]) for i in range(len(val))}
    test_subjects = {int(test[i]["subject_id"]) for i in range(len(test))}
    assert train_subjects.isdisjoint(val_subjects)
    assert val_subjects == set(expected_val)
    assert test_subjects == {11, 12}
    assert train[0]["signal"].shape == (128, 9)
    assert train[0]["features"].shape == (128, 10)
    assert train[0]["target"].dtype == np.int64

    train_signal = np.stack([train[i]["signal"] for i in range(len(train))])
    np.testing.assert_allclose(train_signal.mean(axis=(0, 1)), 0, atol=2e-6)
    np.testing.assert_allclose(train_signal.std(axis=(0, 1)), 1, atol=2e-6)


def test_views_are_deterministic_and_keep_activity_labels(tmp_path):
    root = tmp_path / "har"
    make_extracted_fixture(root)
    prepare_uci_har(root, data_seed=20260716)
    base = UCIHARDataset(root, "test", transform="none")
    reverse = UCIHARDataset(root, "test", transform="reverse")
    shuffle_a = UCIHARDataset(root, "test", transform="shuffle")
    shuffle_b = UCIHARDataset(root, "test", transform="shuffle")
    np.testing.assert_array_equal(reverse[1]["signal"], base[1]["signal"][::-1])
    np.testing.assert_array_equal(shuffle_a[1]["signal"], shuffle_b[1]["signal"])
    for dataset in (reverse, shuffle_a):
        assert dataset[1]["target"] == base[1]["target"]
        assert dataset[1]["base_target"] == base[1]["target"]


def test_gc_targets_and_ood_views_are_causal_and_deterministic(tmp_path):
    root = tmp_path / "har"
    make_extracted_fixture(root)
    prepare_uci_har(root, data_seed=20260716)
    base = UCIHARDataset(root, "test", transform="none")[0]
    reverse = UCIHARDataset(root, "test", transform="reverse")[0]
    prefix = UCIHARDataset(root, "test", transform="prefix50")[0]
    noise_a = UCIHARDataset(root, "test", transform="noise_025")[0]
    noise_b = UCIHARDataset(root, "test", transform="noise_025")[0]

    np.testing.assert_array_equal(prefix["signal"], base["signal"][:64])
    assert prefix["signal"].shape == (64, 9)
    assert prefix["features"].shape == (64, 10)
    assert prefix["coordinate_targets"].shape == (64, 3, 9)
    assert prefix["coordinate_targets"].dtype == np.float32
    assert prefix["coordinate_mask"].shape == (64, 3, 1)
    assert prefix["coordinate_mask"].dtype == np.bool_

    coordinates = reverse["coordinate_targets"]
    np.testing.assert_array_equal(coordinates[:, 0], reverse["signal"])
    np.testing.assert_array_equal(
        coordinates[1:, 1], reverse["signal"][1:] - reverse["signal"][:-1]
    )
    np.testing.assert_array_equal(
        coordinates[2:, 2],
        reverse["signal"][2:]
        - 2 * reverse["signal"][1:-1]
        + reverse["signal"][:-2],
    )
    np.testing.assert_array_equal(coordinates[0, 1:], 0.0)
    np.testing.assert_array_equal(coordinates[1, 2], 0.0)
    assert reverse["coordinate_mask"][:, 0, 0].all()
    assert not reverse["coordinate_mask"][0, 1:, 0].any()
    assert reverse["coordinate_mask"][1:, 1, 0].all()
    assert not reverse["coordinate_mask"][:2, 2, 0].any()
    assert reverse["coordinate_mask"][2:, 2, 0].all()

    np.testing.assert_array_equal(noise_a["signal"], noise_b["signal"])
    for key in ("features", "signal", "coordinate_targets", "coordinate_mask"):
        assert noise_a[key].tobytes() == noise_b[key].tobytes()
    assert not np.array_equal(noise_a["signal"], base["signal"])
    seed_bytes = hashlib.sha256(
        f"20260716:{base['sample_id']}:noise_025".encode("utf-8")
    ).digest()[:8]
    expected_noise = np.random.default_rng(
        int.from_bytes(seed_bytes, "little", signed=False)
    ).normal(0.0, 0.25, size=base["signal"].shape)
    np.testing.assert_array_equal(
        noise_a["signal"], (base["signal"] + expected_noise).astype(np.float32)
    )
    assert noise_a["target"] == base["target"]
    assert noise_a["base_target"] == base["base_target"]


def test_preparation_is_byte_reproducible(tmp_path):
    roots = [tmp_path / "a", tmp_path / "b"]
    manifests = []
    for root in roots:
        make_extracted_fixture(root)
        manifests.append(prepare_uci_har(root, data_seed=20260716))
    assert manifests[0]["manifest_sha256"] == manifests[1]["manifest_sha256"]
    assert manifests[0]["files"] == manifests[1]["files"]
    assert json.loads((roots[0] / "manifest.json").read_text(encoding="utf-8")) == manifests[0]


def test_safe_extraction_rejects_traversal(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="unsafe"):
            _safe_extract_zip(archive, tmp_path / "destination")
    assert not (tmp_path / "escape.txt").exists()


def test_download_manifest_distinguishes_source_from_mirror(tmp_path, monkeypatch):
    mirror = "https://mirror.example/UCI-HAR.zip"

    def fake_download(path, url):
        assert url == mirror
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("UCI HAR Dataset/train/Inertial Signals/", "")
            archive.writestr("UCI HAR Dataset/test/Inertial Signals/", "")

    monkeypatch.setattr(uci_har, "_download_archive", fake_download)
    manifest = download_uci_har(tmp_path / "download", download_url=mirror)
    assert manifest["source_url"] == UCI_HAR_URL
    assert manifest["download_url"] == mirror
