import hashlib
import json

import numpy as np

from temporal_mamba.datasets.temporal_logic import (
    FORMULA_FAMILIES,
    TemporalLogicDataset,
    TemporalQuery,
    build_temporal_logic_manifest,
    encode_query,
    evaluate_query,
)


def test_eventually_window_positive_and_negative():
    x = np.zeros((8, 2), dtype=np.float32)
    x[3, 0] = 1
    q = TemporalQuery("EVENTUALLY", event_a=0, p0=2, p1=5)
    assert evaluate_query(x, q) is True
    x[3, 0] = 0
    x[7, 0] = 1
    assert evaluate_query(x, q) is False


def test_before_and_reversal_change_truth():
    x = np.zeros((8, 3), dtype=np.float32)
    x[1, 0] = 1
    x[6, 1] = 1
    q = TemporalQuery("BEFORE", event_a=0, event_b=1)
    assert evaluate_query(x, q) is True
    assert evaluate_query(x[::-1].copy(), q) is False


def test_until_requires_b_and_continuous_a_prefix():
    x = np.zeros((7, 2), dtype=np.float32)
    x[:4, 0] = 1
    x[4, 1] = 1
    q = TemporalQuery("UNTIL", 0, 1)
    assert evaluate_query(x, q) is True
    x[2, 0] = 0
    assert evaluate_query(x, q) is False
    x[2, 0] = 1
    x[4, 1] = 0
    assert evaluate_query(x, q) is False


def test_bounded_response_checks_every_trigger():
    x = np.zeros((10, 2), dtype=np.float32)
    x[[1, 6], 0] = 1
    x[[3, 9], 1] = 1
    q = TemporalQuery("BOUNDED_RESPONSE", 0, 1, p0=3)
    assert evaluate_query(x, q) is True
    x[9, 1] = 0
    assert evaluate_query(x, q) is False


def test_count_within_uses_inclusive_window_and_threshold():
    x = np.zeros((9, 2), dtype=np.float32)
    x[[2, 4, 6], 0] = 1
    q = TemporalQuery("COUNT_WITHIN", 0, p0=2, p1=6, p2=3)
    assert evaluate_query(x, q) is True
    x[6, 0] = 0
    assert evaluate_query(x, q) is False


def test_gap_uses_first_b_after_first_a():
    x = np.zeros((12, 2), dtype=np.float32)
    x[2, 0] = 1
    x[7, 1] = 1
    q = TemporalQuery("GAP", 0, 1, p0=4, p1=6)
    assert evaluate_query(x, q) is True
    x[7, 1] = 0
    x[10, 1] = 1
    assert evaluate_query(x, q) is False


def test_query_encoding_has_fixed_layout():
    query = TemporalQuery("GAP", 2, 4, p0=4, p1=8, p2=1)
    encoded = encode_query(query, event_dim=8, seq_len=17)
    assert encoded.shape == (6 + 8 + 8 + 3,)
    assert encoded[FORMULA_FAMILIES.index("GAP")] == 1
    assert encoded[6 + 2] == 1
    assert encoded[6 + 8 + 4] == 1
    np.testing.assert_allclose(encoded[-3:], [0.25, 0.5, 0.0625])


def test_manifest_is_balanced_verified_and_reproducible(tmp_path):
    sizes = {"train": 120, "val": 60, "test": 60, "long_test": 60}
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    m1 = build_temporal_logic_manifest(root_a, sizes, data_seed=20260716)
    m2 = build_temporal_logic_manifest(root_b, sizes, data_seed=20260716)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["files"] == m2["files"]
    assert m1["cross_split_duplicates"] == 0
    for split, size in sizes.items():
        path = root_a / f"{split}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m1["files"][split]["sha256"]
        data = np.load(path, allow_pickle=False)
        assert len(data["target"]) == size
        assert data["signal"].shape[1] == (256 if split == "long_test" else 128)
        for family in range(6):
            labels = data["target"][data["family"] == family]
            assert abs(int(labels.sum()) - (len(labels) - int(labels.sum()))) <= 1
    disk_manifest = json.loads((root_a / "manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest == m1


def test_reverse_and_shuffle_recompute_truth_deterministically(tmp_path):
    sizes = {"train": 24, "val": 12, "test": 12, "long_test": 12}
    root = tmp_path / "logic"
    manifest = build_temporal_logic_manifest(root, sizes, data_seed=20260716)
    base = TemporalLogicDataset(root, "train", transform="none")
    reverse = TemporalLogicDataset(root, "train", transform="reverse")
    shuffled_a = TemporalLogicDataset(root, "train", transform="shuffle")
    shuffled_b = TemporalLogicDataset(root, "train", transform="shuffle")
    assert len(base) == sizes["train"]
    assert base[0]["features"].shape == (128, 8 + 1 + 25)
    assert reverse[0]["base_target"] == base[0]["target"]
    np.testing.assert_array_equal(reverse[0]["signal"], base[0]["signal"][::-1])
    np.testing.assert_array_equal(shuffled_a[3]["signal"], shuffled_b[3]["signal"])
    assert shuffled_a[3]["sample_id"] == shuffled_b[3]["sample_id"]
    for dataset in (base, reverse, shuffled_a):
        item = dataset[5]
        query = dataset.query_at(5)
        assert bool(item["target"]) == evaluate_query(item["signal"], query)
    assert base.data_seed == manifest["data_seed"]
