import hashlib

import numpy as np

from temporal_mamba.datasets.temporal_logic import (
    FORMULA_FAMILIES,
    TemporalQuery,
    evaluate_query,
)
from temporal_mamba.datasets.temporal_logic_v2 import (
    V2_SPLITS,
    TemporalLogicV2Dataset,
    build_temporal_logic_v2_manifest,
)


SMALL_SIZES = {split: 120 for split in V2_SPLITS}


def _query_from_arrays(data, index: int) -> TemporalQuery:
    return TemporalQuery(
        family=FORMULA_FAMILIES[int(data["family"][index])],
        event_a=int(data["event_a"][index]),
        event_b=int(data["event_b"][index]),
        p0=int(data["p0"][index]),
        p1=int(data["p1"][index]),
        p2=int(data["p2"][index]),
    )


def _relevant_event_count(data, index: int) -> tuple[int, int]:
    query = _query_from_arrays(data, index)
    signal = data["signal"][index]
    a_count = int(signal[:, query.event_a].sum())
    b_count = 0 if query.event_b < 0 else int(signal[:, query.event_b].sum())
    return a_count, b_count


def _fingerprint(data, index: int) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(data["signal"][index]).tobytes())
    digest.update(
        np.asarray(
            [
                int(data[name][index])
                for name in ("family", "event_a", "event_b", "p0", "p1", "p2", "target")
            ],
            dtype="<i8",
        ).tobytes()
    )
    return digest.hexdigest()


def test_v2_manifest_is_balanced_verified_unique_and_reproducible(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    first = build_temporal_logic_v2_manifest(root_a, SMALL_SIZES, data_seed=20260717)
    second = build_temporal_logic_v2_manifest(root_b, SMALL_SIZES, data_seed=20260717)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["files"] == second["files"]
    assert first["cross_split_duplicates"] == 0
    assert set(first["files"]) == set(V2_SPLITS)

    fingerprints: set[str] = set()
    for split in V2_SPLITS:
        with np.load(root_a / f"{split}.npz", allow_pickle=False) as data:
            expected_length = 256 if split == "long_test" else 128
            assert data["signal"].shape == (SMALL_SIZES[split], expected_length, 8)
            for family in range(len(FORMULA_FAMILIES)):
                labels = data["target"][data["family"] == family]
                assert int((labels == 0).sum()) == int((labels == 1).sum())
            for index in range(len(data["target"])):
                query = _query_from_arrays(data, index)
                assert evaluate_query(data["signal"][index], query) is bool(data["target"][index])
                fingerprint = _fingerprint(data, index)
                assert fingerprint not in fingerprints
                fingerprints.add(fingerprint)


def test_v2_pairs_match_global_relevant_event_counts(tmp_path):
    root = tmp_path / "logic"
    build_temporal_logic_v2_manifest(root, SMALL_SIZES, data_seed=20260717)

    with np.load(root / "train.npz", allow_pickle=False) as data:
        for family in range(len(FORMULA_FAMILIES)):
            subset = np.flatnonzero(data["family"] == family)
            negatives = subset[data["target"][subset] == 0]
            positives = subset[data["target"][subset] == 1]
            negative_counts = sorted(_relevant_event_count(data, int(i)) for i in negatives)
            positive_counts = sorted(_relevant_event_count(data, int(i)) for i in positives)
            assert negative_counts == positive_counts


def test_v2_channel_ood_reserves_event_a_channels(tmp_path):
    root = tmp_path / "logic"
    build_temporal_logic_v2_manifest(root, SMALL_SIZES, data_seed=20260717)

    for split in V2_SPLITS:
        with np.load(root / f"{split}.npz", allow_pickle=False) as data:
            if split == "channel_ood":
                assert set(np.unique(data["event_a"])).issubset({6, 7})
                assert set(np.unique(data["event_a"])) == {6, 7}
            else:
                assert int(data["event_a"].max()) <= 5


def test_v2_transform_keeps_frozen_label_and_structured_query(tmp_path):
    root = tmp_path / "logic"
    build_temporal_logic_v2_manifest(root, SMALL_SIZES, data_seed=20260717)
    base = TemporalLogicV2Dataset(root, "test")
    reverse = TemporalLogicV2Dataset(root, "test", transform="reverse")
    shuffled_a = TemporalLogicV2Dataset(root, "test", transform="shuffle")
    shuffled_b = TemporalLogicV2Dataset(root, "test", transform="shuffle")

    for index in range(len(base)):
        assert reverse[index]["target"] == base[index]["target"]
        assert shuffled_a[index]["target"] == base[index]["target"]
        np.testing.assert_array_equal(reverse[index]["signal"], base[index]["signal"][::-1])
        np.testing.assert_array_equal(shuffled_a[index]["signal"], shuffled_b[index]["signal"])

    item = base[0]
    assert item["query"].shape == (25,)
    assert item["features"].shape == (128, 34)
    assert item["base_target"] == item["target"]
    assert item["formula_family"] in FORMULA_FAMILIES

