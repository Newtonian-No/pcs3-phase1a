# Temporal Logic v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a matched, query-bound temporal-logic v2 benchmark and causal Mamba path that must pass two-seed generalization gates before launching a 12-run ablation matrix.

**Architecture:** Keep v1 and UCI HAR paths unchanged. The v2 dataset persists raw signals plus structured queries; a deterministic binder selects queried A/B streams, a bounded query conditioner modulates shared Mamba layers, and a causal final/max/mean readout classifies the full sequence. Training evaluates ID, long, channel-OOD, reverse-frozen, and shuffle-frozen views and a strict gate blocks the full matrix unless both diagnostic seeds pass.

**Tech Stack:** Python 3.11, NumPy, PyTorch, pytest, JSON/NPZ artifacts, Git, NVIDIA RTX PRO 6000.

## Global Constraints

- Preserve `temporal_logic` v1 datasets, code behavior, `artifacts/full`, and `docs/experiment_report.md`.
- Do not introduce attention, a temporal rule engine, or formula truth values as model inputs.
- Keep the explicit float32 selective recurrence, bounded `dt` in `[1e-3, 1e-1]`, zero-initialized error modulation, and shared encoder parameters.
- Use variants `vanilla`, `two_pass`, `error_inject`, and `error_aux` for the v2 full matrix; use seeds 42, 123, and 777.
- Store v2 data at `/home/lab/datasets/pcs3-temporal/temporal_logic_v2` and artifacts under `artifacts/v2-*`.
- Run every Pro 6000 Python process with CPU affinity `0-7,10-31`.
- Do not launch the 12-run matrix unless seeds 42 and 123 each reach validation balanced accuracy >=0.80, every family >=0.70, and channel-OOD balanced accuracy >=0.70.

---

### Task 1: Temporal-logic v2 configuration and matched dataset

**Files:**
- Create: `temporal_mamba/datasets/temporal_logic_v2.py`
- Create: `configs/temporal_logic_v2.json`
- Create: `configs/temporal_logic_v2_raw.json`
- Modify: `temporal_mamba/config.py`
- Modify: `temporal_mamba/datasets/__init__.py`
- Test: `tests/test_temporal_logic_v2.py`
- Test: `tests/test_temporal_config.py`

**Interfaces:**
- Produces: `V2_SPLITS`, `build_temporal_logic_v2_manifest(root, sizes, data_seed, event_dim, seq_len, long_seq_len) -> dict[str, object]`.
- Produces: `TemporalLogicV2Dataset(root, split, transform="none")` returning `features`, `signal`, `query`, `target`, `base_target`, `sample_id`, and `formula_family`.
- Produces: `ExperimentConfig.input_mode` with values `standard`, `raw_concat`, or `query_bound`.

- [ ] **Step 1: Write failing dataset-construction tests**

Add tests that require equal `(family, label)` counts, no duplicate fingerprints, deterministic hashes, five splits, matched global event counts, frozen transformed labels, and exact query shapes:

```python
def test_v2_manifest_is_balanced_matched_and_reproducible(tmp_path):
    sizes = {name: 120 for name in ("train", "val", "test", "long_test", "channel_ood")}
    first = build_temporal_logic_v2_manifest(tmp_path / "a", sizes, data_seed=20260717)
    second = build_temporal_logic_v2_manifest(tmp_path / "b", sizes, data_seed=20260717)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["cross_split_duplicates"] == 0
    assert set(first["files"]) == set(sizes)
    for split in sizes:
        data = np.load(tmp_path / "a" / f"{split}.npz", allow_pickle=False)
        for family in range(6):
            labels = data["target"][data["family"] == family]
            assert int((labels == 0).sum()) == int((labels == 1).sum())
        for index in range(len(data["target"])):
            signal = data["signal"][index]
            query = query_from_arrays(data, index)
            assert int(evaluate_query(signal, query)) == int(data["target"][index])

def test_v2_pairs_match_global_relevant_event_counts(tmp_path):
    root = tmp_path / "logic"
    build_temporal_logic_v2_manifest(root, {name: 120 for name in V2_SPLITS}, data_seed=20260717)
    data = np.load(root / "train.npz", allow_pickle=False)
    for family in range(6):
        subset = np.flatnonzero(data["family"] == family)
        negatives = subset[data["target"][subset] == 0]
        positives = subset[data["target"][subset] == 1]
        negative_counts = sorted(relevant_event_count(data, int(i)) for i in negatives)
        positive_counts = sorted(relevant_event_count(data, int(i)) for i in positives)
        assert negative_counts == positive_counts

def test_v2_transform_keeps_balanced_frozen_label(tmp_path):
    root = tmp_path / "logic"
    build_temporal_logic_v2_manifest(root, {name: 120 for name in V2_SPLITS}, data_seed=20260717)
    base = TemporalLogicV2Dataset(root, "test")
    reverse = TemporalLogicV2Dataset(root, "test", transform="reverse")
    shuffle = TemporalLogicV2Dataset(root, "test", transform="shuffle")
    for index in range(len(base)):
        assert reverse[index]["target"] == base[index]["target"]
        assert shuffle[index]["target"] == base[index]["target"]
        np.testing.assert_array_equal(reverse[index]["signal"], base[index]["signal"][::-1])
    assert base[0]["query"].shape == (25,)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_temporal_logic_v2.py tests/test_temporal_config.py -q`

Expected: collection/import failure because `temporal_logic_v2` and `input_mode` do not exist.

- [ ] **Step 3: Implement the matched generator and frozen views**

Implement `TemporalQuery` reuse and these exact split/record contracts:

```python
V2_SPLITS = ("train", "val", "test", "long_test", "channel_ood")
DEFAULT_V2_SIZES = {
    "train": 12_000,
    "val": 2_400,
    "test": 2_400,
    "long_test": 2_400,
    "channel_ood": 2_400,
}

def _matched_record(rng, family, label, seq_len, event_dim, *, ood):
    # Select A/B from channels 0..5 for ID and require channel 6 or 7 for OOD.
    allowed = np.arange(6) if not ood else np.arange(6, event_dim)
    event_a = int(rng.choice(allowed))
    other = np.asarray([i for i in range(event_dim) if i != event_a])
    event_b = int(rng.choice(other))
    # Each family changes one temporal relation while preserving global counts:
    # EVENTUALLY moves A across the window boundary; BEFORE swaps A/B order;
    # UNTIL removes one prefix A and compensates after B; BOUNDED_RESPONSE
    # moves one B beyond its response horizon; COUNT_WITHIN moves one A out of
    # the count window; GAP moves B to the nearest invalid gap boundary.
    signal, query = _construct_matched_family(
        rng, family, bool(label), seq_len, event_dim, event_a, event_b
    )
    if evaluate_query(signal, query) is not bool(label):
        raise AssertionError("v2 constructor produced the wrong truth value")
    return signal, query
```

`_construct_matched_family` must implement the following six algorithms. Sample every decisive timestamp away from sequence edges so both labels remain valid. `EVENTUALLY` gives each label the same A count and moves one decisive A from uniformly inside `[p0, p1]` to uniformly outside it. `BEFORE` samples two distinct timestamps and assigns the earlier one to A for positives or B for negatives. `UNTIL` creates an A at every prefix step before the first B for positives; negatives clear one uniformly selected prefix A and add one A after B. `BOUNDED_RESPONSE` uses identical A triggers and B responses except that negatives move one B from `trigger + [p0, p1]` to the nearest valid timestamp beyond that interval. `COUNT_WITHIN` samples exactly `p2` global A events; positives put all `p2` inside `[p0, p1]`, while negatives put `p2 - 1` inside and one outside. `GAP` uses one A and one B; positives sample `B-A` inside `[p0, p1]`, while negatives set the gap to `p0-1` or `p1+1`, choosing a valid nearest boundary. Add matched distractor events with the same distribution for both labels, then verify the query result. Persist the same query arrays as v1 plus `sample_id`, `signal`, and `target`; fingerprint the complete signal/query/target tuple and reject duplicates across all earlier splits.

`TemporalLogicV2Dataset.__getitem__` must freeze `target` before any transform and return legacy concatenated `features` for the raw diagnostic while keeping `signal` and `query` separate:

```python
query_vector = encode_query(query, event_dim=event_dim, seq_len=length)
time = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
legacy = np.concatenate([
    signal,
    time,
    np.broadcast_to(query_vector, (length, len(query_vector))),
], axis=-1).astype(np.float32, copy=False)
return {
    "features": legacy,
    "signal": signal.astype(np.float32, copy=False),
    "query": query_vector,
    "target": base_target,
    "base_target": base_target,
    "sample_id": sample_id,
    "formula_family": query.family,
}
```

Extend config loading so `input_mode` is optional for existing configs and required to be `query_bound` or `raw_concat` for v2. Use `standard` as the legacy default and treat `temporal_logic_v2` as a binary temporal dataset with zero validation fraction.

- [ ] **Step 4: Run dataset/config tests and full regression**

Run: `python -m pytest tests/test_temporal_logic_v2.py tests/test_temporal_config.py -q`

Expected: all new tests pass.

Run: `python -m pytest -q`

Expected: the existing 78 tests plus new tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add temporal_mamba/datasets/temporal_logic_v2.py temporal_mamba/datasets/__init__.py temporal_mamba/config.py configs/temporal_logic_v2.json configs/temporal_logic_v2_raw.json tests/test_temporal_logic_v2.py tests/test_temporal_config.py
git commit -m "feat: add matched temporal logic v2 benchmark"
```

---

### Task 2: Explicit query binder and bounded conditioner

**Files:**
- Create: `temporal_mamba/query_binding.py`
- Create: `tests/test_query_binding.py`

**Interfaces:**
- Produces: `BoundTemporalInput(sequence: Tensor, condition: Tensor, prediction_signal: Tensor)`.
- Produces: `TemporalQueryBinder(event_dim: int, family_dim: int = 6)` with `forward(signal: Tensor, query: Tensor) -> BoundTemporalInput`.
- Produces: `BoundedQueryFiLM(condition_dim: int, n_layers: int, d_model: int, scale_limit: float = 0.25, shift_limit: float = 0.25)` returning `(scale, shift)` tensors shaped `B x L x D`.

- [ ] **Step 1: Write failing binder and invariance tests**

```python
def test_binder_extracts_exact_a_b_and_relative_time():
    signal = torch.zeros(2, 5, 8)
    signal[0, 1, 3] = 1
    signal[0, 4, 6] = 1
    query = make_query(event_a=3, event_b=6, p0=1, p1=4, seq_len=5)
    bound = TemporalQueryBinder(event_dim=8)(signal[:1], query)
    torch.testing.assert_close(bound.sequence[0, :, 0], signal[0, :, 3])
    torch.testing.assert_close(bound.sequence[0, :, 1], signal[0, :, 6])
    torch.testing.assert_close(bound.sequence[0, :, 2], torch.linspace(0, 1, 5))
    assert bound.sequence.shape == (1, 5, 5)
    assert bound.condition.shape == (1, 9)
    assert bound.prediction_signal.shape == (1, 5, 2)

def test_binder_is_invariant_to_joint_channel_permutation():
    signal, query = random_signal_and_query(batch=4, length=17, event_dim=8)
    permutation = torch.tensor([6, 2, 0, 7, 4, 1, 5, 3])
    permuted_signal = signal[:, :, permutation]
    permuted_query = permute_query_channels(query, permutation)
    binder = TemporalQueryBinder(event_dim=8)
    torch.testing.assert_close(
        binder(signal, query).sequence,
        binder(permuted_signal, permuted_query).sequence,
    )

def test_film_starts_identity_and_stays_bounded():
    film = BoundedQueryFiLM(condition_dim=9, n_layers=4, d_model=16)
    scale, shift = film(torch.randn(3, 9) * 1e6)
    assert torch.all(scale >= 0.75) and torch.all(scale <= 1.25)
    assert torch.all(shift >= -0.25) and torch.all(shift <= 0.25)
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(shift, torch.zeros_like(shift))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_query_binding.py -q`

Expected: import failure because `query_binding.py` does not exist.

- [ ] **Step 3: Implement binder and conditioner**

```python
@dataclass(frozen=True)
class BoundTemporalInput:
    sequence: Tensor
    condition: Tensor
    prediction_signal: Tensor

class TemporalQueryBinder(nn.Module):
    def __init__(self, event_dim: int, family_dim: int = 6) -> None:
        super().__init__()
        self.event_dim = event_dim
        self.family_dim = family_dim

    def forward(self, signal: Tensor, query: Tensor) -> BoundTemporalInput:
        family = query[:, :self.family_dim].float()
        a_one_hot = query[:, self.family_dim:self.family_dim + self.event_dim].float()
        b_start = self.family_dim + self.event_dim
        b_one_hot = query[:, b_start:b_start + self.event_dim].float()
        params = query[:, b_start + self.event_dim:b_start + self.event_dim + 3].float()
        a = torch.einsum("btd,bd->bt", signal.float(), a_one_hot)
        b = torch.einsum("btd,bd->bt", signal.float(), b_one_hot)
        time = torch.linspace(0, 1, signal.shape[1], device=signal.device, dtype=torch.float32)
        time = time.expand(signal.shape[0], -1)
        sequence = torch.stack((a, b, time, time - params[:, 0:1], time - params[:, 1:2]), dim=-1)
        return BoundTemporalInput(
            sequence=sequence,
            condition=torch.cat((family, params), dim=-1),
            prediction_signal=torch.stack((a, b), dim=-1),
        )

class BoundedQueryFiLM(nn.Module):
    def __init__(self, condition_dim, n_layers, d_model, scale_limit=0.25, shift_limit=0.25):
        super().__init__()
        self.n_layers, self.d_model = n_layers, d_model
        self.scale_limit, self.shift_limit = scale_limit, shift_limit
        self.projection = nn.Linear(condition_dim, 2 * n_layers * d_model)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, condition):
        raw = self.projection(condition.float()).view(-1, self.n_layers, 2, self.d_model)
        scale = 1.0 + self.scale_limit * torch.tanh(raw[:, :, 0])
        shift = self.shift_limit * torch.tanh(raw[:, :, 1])
        return scale, shift
```

Validate all tensor ranks, dimensions, finite inputs, event one-hot sums, and missing-B all-zero encoding before computing outputs.

- [ ] **Step 4: Run new and full tests**

Run: `python -m pytest tests/test_query_binding.py -q`

Expected: all binder tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add temporal_mamba/query_binding.py tests/test_query_binding.py
git commit -m "feat: add explicit temporal query binding"
```

---

### Task 3: Query-conditioned Mamba, causal readout, and bound error target

**Files:**
- Modify: `temporal_mamba/model.py`
- Modify: `tests/test_temporal_model.py`
- Test: `tests/test_query_binding.py`

**Interfaces:**
- Extends: `TemporalMambaModel(..., input_mode: str = "standard")`.
- Extends: `forward(features, signal, *, variant, query=None, return_diagnostics=False)`.
- Preserves: existing v1/HAR output shapes and exact variant semantics.

- [ ] **Step 1: Write failing model integration tests**

```python
def test_query_bound_model_is_channel_permutation_invariant():
    model = make_tiny_v2_model().eval()
    signal, query = random_signal_and_query(batch=3, length=24, event_dim=8)
    permutation = torch.tensor([3, 7, 1, 6, 0, 5, 2, 4])
    first = model(signal, signal, query=query, variant="vanilla")
    second = model(
        signal[:, :, permutation],
        signal[:, :, permutation],
        query=permute_query_channels(query, permutation),
        variant="vanilla",
    )
    torch.testing.assert_close(first.logits, second.logits)

def test_v2_readout_uses_final_max_and_mean_states():
    model = make_tiny_v2_model().eval()
    assert model.classifier.in_features == 3 * model.model_config.d_model
    output = model(*v2_inputs(batch=2, length=32), variant="error_aux", return_diagnostics=True)
    assert output.logits.shape == (2, 1)
    assert output.position_error.shape == (2, 32, 2)
    assert output.velocity_error.shape == (2, 32, 2)
    assert bool(output.diagnostics["finite"])

def test_legacy_model_contract_is_unchanged():
    model = make_tiny_model().eval()
    output = model(torch.randn(2, 12, 20), torch.randn(2, 12, 4), variant="vanilla")
    assert output.logits.shape == (2, 1)
    assert model.classifier.in_features == model.model_config.d_model
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_temporal_model.py tests/test_query_binding.py -q`

Expected: failure because the model has no `input_mode` or `query` path.

- [ ] **Step 3: Implement minimal model integration**

For `input_mode="query_bound"`, construct `TemporalQueryBinder(signal_dim)`, `BoundedQueryFiLM(9, n_layers, d_model)`, an input projection from 5 streams, a predictor with output dimension 2, and a classifier from `3*d_model`. For all other modes, keep existing dimensions and modules.

Change `_encode` to apply one bounded FiLM pair before each existing layer:

```python
for index, layer in enumerate(self.layers):
    if film is not None:
        scale, shift = film
        hidden = hidden * scale[:, index:index + 1, :] + shift[:, index:index + 1, :]
    hidden = layer(hidden, error=error, return_diagnostics=return_diagnostics)
```

At the forward boundary:

```python
if self.input_mode == "query_bound":
    if query is None:
        raise ValueError("query_bound input requires query")
    bound = self.query_binder(signal, query)
    encoded_features = bound.sequence
    prediction_signal = bound.prediction_signal
    film = self.query_film(bound.condition)
else:
    encoded_features = features
    prediction_signal = signal
    film = None
```

Use `prediction_signal` in `aligned_errors`. After the final output norm, use:

```python
if self.input_mode == "query_bound":
    final = final_hidden[:, -1]
    prefix_max = torch.cummax(final_hidden, dim=1).values[:, -1]
    prefix_sum = torch.cumsum(final_hidden, dim=1)
    denominator = torch.arange(1, final_hidden.shape[1] + 1, device=final_hidden.device)
    prefix_mean = (prefix_sum / denominator[None, :, None])[:, -1]
    readout = torch.cat((final, prefix_max, prefix_mean), dim=-1)
else:
    readout = final_hidden[:, -1]
logits = self.classifier(readout)
```

- [ ] **Step 4: Verify model and regression tests**

Run: `python -m pytest tests/test_temporal_model.py tests/test_query_binding.py tests/test_direct_scan.py -q`

Expected: all selected tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add temporal_mamba/model.py tests/test_temporal_model.py tests/test_query_binding.py
git commit -m "feat: condition causal Mamba on bound queries"
```

---

### Task 4: Training, evaluation views, and v2 artifacts

**Files:**
- Modify: `temporal_mamba/train.py`
- Modify: `temporal_mamba/losses.py`
- Modify: `temporal_mamba/metrics.py`
- Modify: `tests/test_temporal_training.py`
- Modify: `tests/test_temporal_losses_metrics.py`

**Interfaces:**
- Extends: batch movement to return optional `query` without changing legacy callers.
- Produces final metrics keys: `val`, `test`, `long_test`, `channel_ood`, `reverse_frozen`, `shuffle_frozen`.

- [ ] **Step 1: Write failing v2 training/evaluation tests**

```python
def test_build_v2_datasets_has_all_required_views(tmp_path, v2_config):
    datasets = build_datasets(v2_config, tmp_path / "data")
    assert set(datasets) == {"train", "val", "test", "long_test", "channel_ood"}

def test_move_batch_preserves_optional_query():
    moved = _move_batch({
        "features": torch.randn(2, 8, 34),
        "signal": torch.randn(2, 8, 8),
        "query": torch.randn(2, 25),
        "target": torch.tensor([0.0, 1.0]),
    }, torch.device("cpu"))
    assert moved.query.shape == (2, 25)

def test_v2_final_contains_ood_and_frozen_control_metrics(tmp_path, monkeypatch):
    final = run_one_epoch_v2(tmp_path, monkeypatch)
    assert set(final["metrics"]) == {
        "val", "test", "long_test", "channel_ood", "reverse_frozen", "shuffle_frozen"
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_temporal_training.py tests/test_temporal_losses_metrics.py -q`

Expected: failure because v2 is unknown and query/control views are not passed.

- [ ] **Step 3: Implement v2 training plumbing**

Add a frozen `BatchTensors` dataclass with `features`, `signal`, `target`, and `query`. `_move_batch` returns it; every train/evaluate/overfit call passes `query=batch.query` into the model. Treat both temporal datasets as binary in `_predictions`, `compute_task_loss`, and metric selection.

For v2, `_input_dim` returns 5 in `query_bound` mode and the legacy concatenated dimension in `raw_concat` mode. Construct `TemporalMambaModel(input_mode=config.input_mode, ...)`.

After loading the best checkpoint, evaluate the five persisted splits plus two test views:

```python
reverse_loader = DataLoader(
    TemporalLogicV2Dataset(data_root, "test", transform="reverse"),
    batch_size=config.training.batch_size,
    shuffle=False,
)
shuffle_loader = DataLoader(
    TemporalLogicV2Dataset(data_root, "test", transform="shuffle"),
    batch_size=config.training.batch_size,
    shuffle=False,
)
metrics = {
    "val": evaluate(model, loaders["val"], config, device=resolved_device),
    "test": evaluate(model, loaders["test"], config, device=resolved_device),
    "long_test": evaluate(model, loaders["long_test"], config, device=resolved_device),
    "channel_ood": evaluate(model, loaders["channel_ood"], config, device=resolved_device),
    "reverse_frozen": evaluate(model, reverse_loader, config, device=resolved_device),
    "shuffle_frozen": evaluate(model, shuffle_loader, config, device=resolved_device),
}
```

Keep the legacy final-artifact schema for v1/HAR and bump v2 finals to schema version 2.

- [ ] **Step 4: Run training and full regression tests**

Run: `python -m pytest tests/test_temporal_training.py tests/test_temporal_losses_metrics.py -q`

Expected: selected tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add temporal_mamba/train.py temporal_mamba/losses.py temporal_mamba/metrics.py tests/test_temporal_training.py tests/test_temporal_losses_metrics.py
git commit -m "feat: train and evaluate temporal logic v2"
```

---

### Task 5: Gate, matrix, strict summary, and attribution report

**Files:**
- Create: `temporal_mamba/v2_gate.py`
- Create: `temporal_mamba/summarize_v2.py`
- Modify: `temporal_mamba/run_matrix.py`
- Create: `tests/test_v2_gate_summary.py`
- Modify: `tests/test_temporal_matrix_summary.py`

**Interfaces:**
- Produces: `validate_v2_gate(artifact_root, seeds=(42, 123), overall=0.80, family=0.70, ood=0.70) -> dict`.
- Produces: `summarize_v2_matrix(artifact_root, report_path) -> dict` requiring exactly 12 verified finals.

- [ ] **Step 1: Write failing gate and summary tests**

```python
def test_v2_gate_requires_every_seed_and_family(tmp_path):
    write_v2_final(tmp_path, seed=42, overall=0.82, family_min=0.72, ood=0.75)
    write_v2_final(tmp_path, seed=123, overall=0.83, family_min=0.71, ood=0.74)
    result = validate_v2_gate(tmp_path)
    assert result["passed"] is True
    overwrite_family_score(tmp_path, seed=123, family="GAP", score=0.69)
    with pytest.raises(ValueError, match="GAP"):
        validate_v2_gate(tmp_path)

def test_v2_summary_requires_12_matching_finals(tmp_path):
    write_complete_v2_matrix(tmp_path, commit="abc", manifest="data")
    summary = summarize_v2_matrix(tmp_path, report_path=tmp_path / "report.md")
    assert summary["validation"]["complete_runs"] == 12
    assert "binder_minus_raw" in summary["attribution"]
    (tmp_path / "temporal_logic_v2-error_aux-seed777" / "final.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        summarize_v2_matrix(tmp_path, report_path=tmp_path / "report.md")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_v2_gate_summary.py tests/test_temporal_matrix_summary.py -q`

Expected: import failure because gate/summary modules do not exist.

- [ ] **Step 3: Implement strict gates and v2 matrix expansion**

`validate_v2_gate` loads bound-vanilla finals for both seeds, checks status/identity/hash/finite fields, validates overall, each of six `per_family` scores, and channel-OOD. It writes `gate.json` atomically only after both seeds pass.

`run_matrix` accepts dataset `temporal_logic_v2` and explicit variants. The v2 full command expands exactly:

```python
[
    RunSpec("temporal_logic_v2", variant, seed)
    for variant in ("vanilla", "two_pass", "error_inject", "error_aux")
    for seed in (42, 123, 777)
]
```

`summarize_v2_matrix` reuses finite/hash validation, aggregates all six evaluation views and formula families, calculates paired deltas, reads the raw diagnostic final from `artifacts/v2-smoke`, and writes `summary.json`, `summary.csv`, and a Markdown report with explicit gate status and order-control interpretation.

- [ ] **Step 4: Run selected and full tests**

Run: `python -m pytest tests/test_v2_gate_summary.py tests/test_temporal_matrix_summary.py -q`

Expected: selected tests pass.

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add temporal_mamba/v2_gate.py temporal_mamba/summarize_v2.py temporal_mamba/run_matrix.py tests/test_v2_gate_summary.py tests/test_temporal_matrix_summary.py
git commit -m "feat: gate and summarize temporal logic v2 matrix"
```

---

### Task 6: Pro 6000 deployment, staged gates, and experiment launch

**Files:**
- Modify after successful matrix: `docs/temporal_logic_v2_report.md`
- Runtime artifacts only: `/home/lab/projects/pcs3-temporal/artifacts/v2-*`

**Interfaces:**
- Consumes: committed feature branch, v2 configs, gate CLI, matrix CLI, summary CLI.
- Produces: verified dataset manifest, smoke artifacts, two-seed gate, 12-run matrix, and report.

- [ ] **Step 1: Push the feature branch and synchronize the clean Pro 6000 clone**

Run locally:

```bash
git push -u origin codex/temporal-logic-v2
```

Run remotely without touching v1 artifacts:

```bash
cd /home/lab/projects/pcs3-temporal
git fetch origin codex/temporal-logic-v2:refs/remotes/origin/codex/temporal-logic-v2
git switch codex/temporal-logic-v2
git merge --ff-only origin/codex/temporal-logic-v2
```

Expected: remote HEAD equals the pushed feature commit and `git status --short` is empty apart from ignored artifacts.

- [ ] **Step 2: Run fresh full tests under safe affinity**

Run:

```bash
taskset -c 0-7,10-31 env PYTHONPYCACHEPREFIX=/tmp/pcs3-v2-pycache \
  /home/lab/miniconda3/envs/evuav/bin/python -m pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Generate and validate the immutable v2 dataset**

Run:

```bash
taskset -c 0-7,10-31 /home/lab/miniconda3/envs/evuav/bin/python \
  -m temporal_mamba.datasets.temporal_logic_v2 \
  --root /home/lab/datasets/pcs3-temporal/temporal_logic_v2
```

Expected: five split files, no duplicates, balanced family-label counts, and a manifest SHA-256.

- [ ] **Step 4: Run raw diagnostic, bound tiny-overfit gate, and HAR regression smoke**

Run raw and bound diagnostics into `artifacts/v2-smoke`, always with safe affinity and CUDA. Run the bound `--overfit-only` command and require >=0.98 accuracy. Run UCI HAR vanilla seed 42 for one epoch into `artifacts/v2-smoke-har` and require finite output.

Expected: raw diagnostic retained for attribution, bound tiny gate passes, HAR smoke exits 0, and all `dt` values remain bounded.

- [ ] **Step 5: Run both Stage 2 seeds and validate the gate**

Run bound vanilla seeds 42 and 123 into `artifacts/v2-gate`, then:

```bash
taskset -c 0-7,10-31 /home/lab/miniconda3/envs/evuav/bin/python \
  -m temporal_mamba.v2_gate --artifact-root artifacts/v2-gate
```

Expected: exit 0 and `gate.json` with `passed: true`. If it exits nonzero, stop and preserve artifacts; do not execute Step 6.

- [ ] **Step 6: Launch and monitor the 12-run full matrix only after gate success**

Run:

```bash
taskset -c 0-7,10-31 env PYTHONPYCACHEPREFIX=/tmp/pcs3-v2-pycache \
  /home/lab/miniconda3/envs/evuav/bin/python -m temporal_mamba.run_matrix \
  --datasets temporal_logic_v2 \
  --variants vanilla two_pass error_inject error_aux \
  --seeds 42 123 777 \
  --artifact-root artifacts/v2-full \
  --data-root /home/lab/datasets/pcs3-temporal \
  --device cuda
```

Expected: 12 `end` events with exit code 0. Monitor process liveness and status at least once per minute; never relaunch a completed matching run.

- [ ] **Step 7: Strictly summarize, sync the report, and verify provenance**

Run `python -m temporal_mamba.summarize_v2 --artifact-root artifacts/v2-full --raw-artifact-root artifacts/v2-smoke --report-path docs/temporal_logic_v2_report.md`, verify 12/12 and all hashes, copy the report locally, run `git diff --check`, and rerun the full test suite before committing the report.

- [ ] **Step 8: Commit the verified report**

```bash
git add docs/temporal_logic_v2_report.md
git commit -m "docs: add temporal logic v2 experiment report"
```
