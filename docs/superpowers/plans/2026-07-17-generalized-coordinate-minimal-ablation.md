# Generalized-Coordinate Minimal Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameter-matched K=0/1/2/3 generalized-coordinate prediction-error ablation and run its preregistered synthetic-dynamics and UCI HAR gates exclusively on the Pro 6000.

**Architecture:** Preserve the existing causal shared-weight two-pass Temporal Mamba and legacy experiment behavior. Add an isolated K=3 predictor and coordinate-target contract used only by generalized-coordinate variants; activated orders are selected by masks, and aligned errors modulate only the bounded second-pass `dt` path. Add deterministic continuous-dynamics data, UCI HAR prefix/noise views, a dedicated matrix runner, and a strict decision summarizer.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, standard-library JSON/hash/statistics/subprocess, Git, SSH, NVIDIA Pro 6000.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-07-17-generalized-coordinate-minimal-ablation-design.md`.
- All training, tiny-batch smoke, ablations, OOD evaluation, and statistical reproduction run only on `lab@100.68.12.20`.
- Local work is limited to editing, static checks, and unit tests; local outputs never enter the scientific report.
- Pro 6000 jobs exclude logical CPUs 8 and 9.
- Existing temporal-logic v1/v2 behavior, manifests, artifacts, variant names, and default matrix remain unchanged.
- The error path may modulate only bounded `dt`; do not add B/C/A, attention, bidirectional, or iterative-PCN paths.
- Every generalized-coordinate model instantiates the same K=3 predictor and adapter; order masks alone select K=0/1/2/3.
- Screening seeds are exactly `42, 123, 777`; confirmation seeds are exactly `42, 123, 777, 2026, 31415`.
- Baseline tag: `checkpoint/query-mamba-v2-before-gc-v1`.
- Feature branch: `exp/generalized-coordinate-minimal-ablation`.
- Failed-hypothesis tag: `checkpoint/gc-v1-terminated`.
- Remote data root: `/home/lab/datasets/pcs3-generalized-coordinate-v1`.
- Remote artifact roots: `artifacts/gc-v1-smoke`, `artifacts/gc-v1-screen`, `artifacts/gc-v1-confirm`.

## File Map

- Create `temporal_mamba/generalized_coordinates.py`: coordinate targets, masks, K=3 predictor, aligned errors, deterministic shuffled/noise controls.
- Create `temporal_mamba/datasets/generalized_dynamics.py`: deterministic analytic continuous-dynamics generation, manifests, and dataset views.
- Create `temporal_mamba/run_gc_matrix.py`: isolated generalized-coordinate matrix expansion and execution.
- Create `temporal_mamba/summarize_gc.py`: strict artifact validation, paired statistics, preregistered decision.
- Create `configs/generalized_dynamics_gc.json`: synthetic-dynamics training configuration.
- Create `configs/uci_har_gc.json`: UCI HAR generalized-coordinate configuration.
- Modify `temporal_mamba/config.py`: dataset/variant contracts without changing legacy `VARIANTS`.
- Modify `temporal_mamba/model.py`: optional generalized-coordinate path; legacy path remains byte-behavior compatible.
- Modify `temporal_mamba/losses.py`: masked order-wise auxiliary prediction loss.
- Modify `temporal_mamba/metrics.py`: expose multiclass balanced accuracy for the synthetic primary metric.
- Modify `temporal_mamba/datasets/__init__.py`: lazy exports for the new dataset.
- Modify `temporal_mamba/datasets/uci_har.py`: causal coordinate targets plus deterministic `prefix50` and `noise_025` views.
- Modify `temporal_mamba/train.py`: optional coordinate tensors, generalized datasets/views, diagnostics and artifact schema v3.
- Add focused tests under `tests/` for each new unit; update existing constructor fixtures only where the optional output contract requires it.

---

### Task 1: Freeze the baseline checkpoint and isolate the branch

**Files:**
- Verify: `docs/temporal_logic_v2_report.md`
- Verify: `docs/superpowers/specs/2026-07-17-generalized-coordinate-minimal-ablation-design.md`

**Interfaces:**
- Consumes: clean `master` containing design commit `cce5dc3`.
- Produces: annotated baseline tag and isolated feature branch.

- [ ] **Step 1: Verify the local baseline is clean and record the exact commit**

Run:

```powershell
git -c safe.directory=C:/Users/wflps/Documents/Codex/2026-07-16/zhe/work/pcs3-github status --short --branch
git -c safe.directory=C:/Users/wflps/Documents/Codex/2026-07-16/zhe/work/pcs3-github rev-parse HEAD
```

Expected: no changed paths; branch is `master`; HEAD contains the approved spec and this plan.

- [ ] **Step 2: Create the immutable baseline tag**

Run:

```powershell
git -c safe.directory=C:/Users/wflps/Documents/Codex/2026-07-16/zhe/work/pcs3-github tag -a checkpoint/query-mamba-v2-before-gc-v1 -m "Query-conditioned Temporal Mamba v2 baseline before generalized-coordinate ablation"
git -c safe.directory=C:/Users/wflps/Documents/Codex/2026-07-16/zhe/work/pcs3-github show --no-patch --decorate checkpoint/query-mamba-v2-before-gc-v1
```

Expected: the tag resolves to the clean pre-implementation commit.

- [ ] **Step 3: Create the experiment branch**

Run:

```powershell
git -c safe.directory=C:/Users/wflps/Documents/Codex/2026-07-16/zhe/work/pcs3-github switch -c exp/generalized-coordinate-minimal-ablation
```

Expected: `Switched to a new branch 'exp/generalized-coordinate-minimal-ablation'`.

- [ ] **Step 4: Record the existing v2 provenance without modifying it**

Run:

```powershell
rg -n "Git|manifest|12/12|balanced" docs/temporal_logic_v2_report.md
```

Expected: the report exposes the completed matrix, Git commit and manifest hash needed for rollback comparison.

---

### Task 2: Add dataset and variant configuration contracts

**Files:**
- Modify: `temporal_mamba/config.py`
- Test: `tests/test_temporal_config.py`

**Interfaces:**
- Consumes: existing `VARIANTS`, `DATASETS`, and `ExperimentConfig`.
- Produces: `GC_VARIANTS`, `GC_MATRIX_VARIANTS`, `GC_SEEDS`, `GC_CONFIRM_SEEDS`, optional top-level field `generalized_coordinates`, `ExperimentConfig.gc_order`, `uses_gc`, `uses_gc_aux`, and validation for `generalized_dynamics`.

- [ ] **Step 1: Write failing configuration tests**

Add these assertions to `tests/test_temporal_config.py` using the file's existing valid-config helper:

```python
from temporal_mamba.config import (
    GC_CONFIRM_SEEDS,
    GC_MATRIX_VARIANTS,
    GC_SEEDS,
    GC_VARIANTS,
    load_experiment_config,
)


def test_gc_contract_constants_are_preregistered():
    assert GC_VARIANTS == (
        "gc_k1",
        "gc_k2",
        "gc_k3",
        "gc_k3_shuffled",
        "gc_k3_noise",
    )
    assert GC_MATRIX_VARIANTS == ("vanilla", "two_pass") + GC_VARIANTS
    assert GC_SEEDS == (42, 123, 777)
    assert GC_CONFIRM_SEEDS == (42, 123, 777, 2026, 31415)


def test_generalized_dynamics_gc_orders(tmp_path):
    raw = _valid_config()
    raw.update(dataset="generalized_dynamics", signal_dim=6, num_outputs=3,
               generalized_coordinates=True)
    path = tmp_path / "gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    expected = {"gc_k1": 1, "gc_k2": 2, "gc_k3": 3,
                "gc_k3_shuffled": 3, "gc_k3_noise": 3}
    for variant, order in expected.items():
        config = load_experiment_config(path, variant=variant, seed=42)
        assert config.uses_gc is True
        assert config.uses_gc_aux is True
        assert config.gc_order == order


def test_gc_baselines_enable_same_modules(tmp_path):
    raw = _valid_config()
    raw.update(dataset="uci_har", signal_dim=9, num_outputs=6,
               generalized_coordinates=True)
    raw["data"]["validation_fraction"] = 0.2
    path = tmp_path / "uci-gc.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_experiment_config(path, variant="vanilla", seed=42).uses_gc is True
    assert load_experiment_config(path, variant="two_pass", seed=42).uses_gc is True


def test_legacy_variants_remain_unchanged():
    from temporal_mamba.config import VARIANTS
    assert VARIANTS == (
        "vanilla", "two_pass", "error_inject", "error_aux",
        "time_shuffle", "time_reverse",
    )
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
pytest tests/test_temporal_config.py -q
```

Expected: collection fails because the GC constants and properties do not exist.

- [ ] **Step 3: Implement the isolated contracts**

Add without changing legacy `VARIANTS`:

```python
GC_VARIANTS = (
    "gc_k1",
    "gc_k2",
    "gc_k3",
    "gc_k3_shuffled",
    "gc_k3_noise",
)
GC_MATRIX_VARIANTS = ("vanilla", "two_pass") + GC_VARIANTS
GC_SEEDS = (42, 123, 777)
GC_CONFIRM_SEEDS = (42, 123, 777, 2026, 31415)
SUPPORTED_VARIANTS = VARIANTS + GC_VARIANTS
DATASETS = ("temporal_logic", "temporal_logic_v2", "uci_har", "generalized_dynamics")
```

Add properties:

```python
@property
def uses_gc(self) -> bool:
    return self.generalized_coordinates

@property
def uses_gc_aux(self) -> bool:
    return self.variant in GC_VARIANTS

@property
def gc_order(self) -> int:
    return {
        "gc_k1": 1,
        "gc_k2": 2,
        "gc_k3": 3,
        "gc_k3_shuffled": 3,
        "gc_k3_noise": 3,
    }.get(self.variant, 0)
```

Add `generalized_coordinates: bool = False` to `ExperimentConfig` and accept it as an optional strict top-level JSON field. Validate variants with `SUPPORTED_VARIANTS`. When the field is true, permit only `vanilla`, `two_pass`, and `GC_VARIANTS`, and only for `generalized_dynamics` or `uci_har`; when false, reject GC variants. Include GC variants in `uses_error` and `uses_aux` while preserving all six legacy results. Retain the v2 restriction to its existing variants. Validate seeds against `GC_CONFIRM_SEEDS`; keep legacy `TRAINING_SEEDS == (42, 123, 777)` unchanged so existing default matrices do not expand silently.

- [ ] **Step 4: Run configuration and legacy tests**

Run:

```powershell
pytest tests/test_temporal_config.py tests/test_temporal_matrix_summary.py tests/test_v2_gate_summary.py -q
```

Expected: all tests pass and legacy variant expectations remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add temporal_mamba/config.py tests/test_temporal_config.py
git commit -m "feat: add generalized-coordinate experiment contracts"
```

---

### Task 3: Implement causal coordinate targets and deterministic controls

**Files:**
- Create: `temporal_mamba/generalized_coordinates.py`
- Create: `tests/test_generalized_coordinates.py`

**Interfaces:**
- Produces: `CoordinateBatch(targets: Tensor, mask: Tensor)`, `causal_coordinate_targets(signal: Tensor)`, `GeneralizedCoordinatePredictor`, `aligned_coordinate_errors(...)`, `select_active_orders(...)`, and `controlled_error(...)`.
- Tensor contract: targets/errors are `B x T x 3 x D`; masks are broadcastable `B x T x 3 x 1`.

- [ ] **Step 1: Write failing target and mask tests**

Create `tests/test_generalized_coordinates.py`:

```python
import torch

from temporal_mamba.generalized_coordinates import causal_coordinate_targets


def test_causal_coordinates_use_only_present_and_past():
    signal = torch.tensor([[[0.0], [1.0], [4.0], [9.0]]])
    batch = causal_coordinate_targets(signal)
    assert batch.targets.shape == (1, 4, 3, 1)
    torch.testing.assert_close(batch.targets[0, :, 0, 0], torch.tensor([0., 1., 4., 9.]))
    torch.testing.assert_close(batch.targets[0, :, 1, 0], torch.tensor([0., 1., 3., 5.]))
    torch.testing.assert_close(batch.targets[0, :, 2, 0], torch.tensor([0., 0., 2., 2.]))
    assert batch.mask[0, :, 0, 0].tolist() == [True, True, True, True]
    assert batch.mask[0, :, 1, 0].tolist() == [False, True, True, True]
    assert batch.mask[0, :, 2, 0].tolist() == [False, False, True, True]


def test_future_mutation_does_not_change_past_coordinates():
    first = torch.randn(2, 8, 3)
    second = first.clone()
    second[:, 6:] += 100
    a = causal_coordinate_targets(first)
    b = causal_coordinate_targets(second)
    torch.testing.assert_close(a.targets[:, :6], b.targets[:, :6])
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_generalized_coordinates.py -q`

Expected: import fails because the module does not exist.

- [ ] **Step 3: Implement coordinate construction**

Use float32 backward differences and explicit masks:

```python
@dataclass(frozen=True)
class CoordinateBatch:
    targets: Tensor
    mask: Tensor


def causal_coordinate_targets(signal: Tensor) -> CoordinateBatch:
    if signal.ndim != 3:
        raise ValueError("signal must be B x T x D")
    x = signal.float()
    dx = torch.zeros_like(x)
    ddx = torch.zeros_like(x)
    dx[:, 1:] = x[:, 1:] - x[:, :-1]
    ddx[:, 2:] = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    targets = torch.stack((x, dx, ddx), dim=2)
    mask = torch.ones((*x.shape[:2], 3, 1), device=x.device, dtype=torch.bool)
    mask[:, 0, 1:] = False
    if x.shape[1] > 1:
        mask[:, 1, 2] = False
    return CoordinateBatch(targets=targets, mask=mask)
```

- [ ] **Step 4: Write failing predictor/alignment/control tests**

Add:

```python
from temporal_mamba.generalized_coordinates import (
    GeneralizedCoordinatePredictor,
    aligned_coordinate_errors,
    controlled_error,
    select_active_orders,
)


def test_aligned_errors_shift_predictions_one_step():
    predictor = GeneralizedCoordinatePredictor(hidden_dim=4, signal_dim=1)
    hidden = torch.randn(2, 5, 4)
    coordinates = causal_coordinate_targets(torch.randn(2, 5, 1))
    errors, valid = aligned_coordinate_errors(hidden, coordinates, predictor)
    assert errors.shape == (2, 5, 3, 1)
    assert not valid[:, 0].any()
    assert valid[:, 1, 0].all()
    assert not valid[:, 1, 2].any()


def test_order_mask_keeps_fixed_flat_dimension():
    errors = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).view(2, 5, 3, 4)
    for order in (1, 2, 3):
        flat = select_active_orders(errors, order)
        assert flat.shape == (2, 5, 12)
        assert torch.count_nonzero(flat[..., order * 4:]) == 0


def test_controls_are_deterministic_and_match_statistics():
    error = torch.randn(8, 9, 12)
    shuffled_a = controlled_error(error, "gc_k3_shuffled", seed=17)
    shuffled_b = controlled_error(error, "gc_k3_shuffled", seed=17)
    torch.testing.assert_close(shuffled_a, shuffled_b)
    torch.testing.assert_close(shuffled_a.mean((0, 1)), error.mean((0, 1)))
    noise = controlled_error(error, "gc_k3_noise", seed=17)
    torch.testing.assert_close(noise.mean((0, 1)), error.mean((0, 1)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(noise.std((0, 1)), error.std((0, 1)), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 5: Implement predictor, alignment, masking and controls**

Implement three same-width heads after one shared RMS-normalized hidden input. Detach hidden and targets at the predictor boundary. Set aligned time index 0 invalid, combine with the supplied coordinate mask, normalize each active order independently over valid batch/time/channel values, keep the flattened error width fixed at `3 * D`, and zero inactive/invalid entries. For shuffled control use a seeded batch permutation; for noise standardize seeded Gaussian values per flattened channel and rescale to the observed error mean/std.

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_generalized_coordinates.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add temporal_mamba/generalized_coordinates.py tests/test_generalized_coordinates.py
git commit -m "feat: add causal generalized-coordinate errors"
```

---

### Task 4: Add deterministic analytic dynamics data

**Files:**
- Create: `temporal_mamba/datasets/generalized_dynamics.py`
- Modify: `temporal_mamba/datasets/__init__.py`
- Create: `tests/test_generalized_dynamics.py`

**Interfaces:**
- Produces: `build_generalized_dynamics_manifest(root, data_seed, sizes, signal_dim=6, seq_len=128)` and `GeneralizedDynamicsDataset(root, split)`.
- Dataset item keys: `features`, `signal`, `coordinate_targets`, `coordinate_mask`, `target`, `base_target`, `sample_id`, `formula_family`.
- Splits: `train`, `val`, `test`, `length_256`, `length_512`, `parameter_ood`, `noise_ood`.

- [ ] **Step 1: Write failing reproducibility and derivative tests**

Create tests that build two small roots with the same seed and assert identical manifest hashes, disjoint sample IDs, balanced three-class labels, shapes, and analytic derivatives:

```python
def test_manifest_is_reproducible_balanced_and_disjoint(tmp_path):
    sizes = {name: 12 for name in (
        "train", "val", "test", "length_256", "length_512", "parameter_ood", "noise_ood"
    )}
    a = build_generalized_dynamics_manifest(tmp_path / "a", 20260717, sizes, signal_dim=3)
    b = build_generalized_dynamics_manifest(tmp_path / "b", 20260717, sizes, signal_dim=3)
    assert a["manifest_sha256"] == b["manifest_sha256"]
    datasets = {split: GeneralizedDynamicsDataset(tmp_path / "a", split) for split in sizes}
    ids = [{datasets[s][i]["sample_id"] for i in range(len(datasets[s]))} for s in sizes]
    assert sum(len(group) for group in ids) == len(set().union(*ids))
    assert np.bincount([datasets["train"][i]["target"] for i in range(12)]).tolist() == [4, 4, 4]


def test_analytic_coordinates_satisfy_signal_contract(tmp_path):
    sizes = {name: 6 for name in (
        "train", "val", "test", "length_256", "length_512", "parameter_ood", "noise_ood"
    )}
    build_generalized_dynamics_manifest(tmp_path, 20260717, sizes, signal_dim=2)
    item = GeneralizedDynamicsDataset(tmp_path, "train")[0]
    assert item["signal"].shape == (128, 2)
    assert item["coordinate_targets"].shape == (128, 3, 2)
    assert item["coordinate_mask"].shape == (128, 3, 1)
    assert item["features"].shape == (128, 3)
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_generalized_dynamics.py -q`

Expected: import fails because the dataset module does not exist.

- [ ] **Step 3: Implement the generators**

Use float64 generation followed by float32 persistence. For each channel sample independent amplitude/phase parameters from the split-specific seeded generator:

```python
def damped(t, amp, phase, damping, omega):
    angle = omega * t + phase
    decay = np.exp(-damping * t)
    x = amp * decay * np.cos(angle)
    dx = amp * decay * (-damping * np.cos(angle) - omega * np.sin(angle))
    ddx = -2 * damping * dx - (damping**2 + omega**2) * x
    return x, dx, ddx


def forced(t, amp, phase, omega, drive_omega):
    x = amp * np.cos(omega * t + phase) + 0.5 * amp * np.cos(drive_omega * t - phase)
    dx = -amp * omega * np.sin(omega * t + phase) - 0.5 * amp * drive_omega * np.sin(drive_omega * t - phase)
    ddx = -amp * omega**2 * np.cos(omega * t + phase) - 0.5 * amp * drive_omega**2 * np.cos(drive_omega * t - phase)
    return x, dx, ddx
```

Implement the switching family as two analytic sinusoidal segments with the second segment initialized from the first segment's switch value and phase, and record the switch rate/position in metadata. ID parameters stay inside fixed ranges; `parameter_ood` samples damping/frequency/switch rate outside them; `noise_ood` adds seeded standardized observation noise only to `signal`, while preserving clean analytic targets for diagnostics. Normalize each signal channel with training-only mean/std and transform analytic derivatives by the same signal scale.

Use this causal piecewise analytic construction:

```python
def switching(t, amp, phase, omega_before, omega_after, switch_index):
    switch_t = t[switch_index]
    angle_before = omega_before * t + phase
    switch_angle = omega_before * switch_t + phase
    angle_after = switch_angle + omega_after * (t - switch_t)
    before = np.arange(len(t)) < switch_index
    angle = np.where(before, angle_before, angle_after)
    omega = np.where(before, omega_before, omega_after)
    x = amp * np.cos(angle)
    dx = -amp * omega * np.sin(angle)
    ddx = -amp * omega**2 * np.cos(angle)
    return x, dx, ddx
```

- [ ] **Step 4: Persist strict manifests and dataset items**

Reuse `_write_npz_atomic`, `_write_json_atomic`, and `_sha256_file`. Manifest schema must contain generator version, ranges, sizes, shapes, split file hashes, normalization statistics, label counts, and final canonical `manifest_sha256`. Reject non-multiples of three because balanced generation is required.

Add an argparse entry point accepting `--root`, `--config`, and optional `--data-seed`; load sizes, signal dimension and sequence length from the strict experiment config, call the builder, and print sorted JSON. This is the exact CLI used on Pro 6000 in Task 10.

- [ ] **Step 5: Export lazily and run tests**

Add the two public names to `temporal_mamba/datasets/__init__.py`, then run:

```powershell
pytest tests/test_generalized_dynamics.py tests/test_temporal_logic.py tests/test_temporal_logic_v2.py tests/test_uci_har.py -q
```

Expected: all dataset tests pass.

- [ ] **Step 6: Commit**

```powershell
git add temporal_mamba/datasets/generalized_dynamics.py temporal_mamba/datasets/__init__.py tests/test_generalized_dynamics.py
git commit -m "feat: add analytic generalized-dynamics dataset"
```

---

### Task 5: Integrate K=3 errors into the model and loss

**Files:**
- Modify: `temporal_mamba/model.py`
- Modify: `temporal_mamba/losses.py`
- Modify: `temporal_mamba/metrics.py`
- Modify: `tests/test_temporal_model.py`
- Modify: `tests/test_temporal_losses_metrics.py`

**Interfaces:**
- `TemporalMambaModel(..., generalized_coordinates: bool = False)`.
- `forward(..., coordinate_targets: Tensor | None = None, coordinate_mask: Tensor | None = None, error_control_seed: int | None = None)`.
- `TemporalModelOutput` adds optional `coordinate_errors`, `coordinate_mask`, and `gc_order` fields with backward-compatible defaults.

- [ ] **Step 1: Write failing parameter-match and forward-contract tests**

Add tests:

```python
@pytest.mark.parametrize("variant,order", [
    ("gc_k1", 1), ("gc_k2", 2), ("gc_k3", 3),
    ("gc_k3_shuffled", 3), ("gc_k3_noise", 3),
])
def test_gc_forward_has_fixed_error_width(variant, order):
    config = ModelConfig(d_model=8, d_state=4, n_layers=2, expand=1,
                         dt_min=1e-3, dt_max=1e-1,
                         alpha_max=1.38629436112, dropout=0.0)
    model = TemporalMambaModel(
        input_dim=7, signal_dim=6, num_outputs=3,
        model_config=config, generalized_coordinates=True,
    )
    signal = torch.randn(4, 16, 6)
    features = torch.cat((signal, torch.linspace(0, 1, 16).view(1, 16, 1).expand(4, -1, -1)), -1)
    coordinates = causal_coordinate_targets(signal)
    output = model(features, signal, variant=variant,
                   coordinate_targets=coordinates.targets,
                   coordinate_mask=coordinates.mask,
                   error_control_seed=99)
    assert output.coordinate_errors.shape == (4, 16, 18)
    assert output.gc_order == order
    assert output.pass_count == 2


def test_all_gc_variants_have_identical_parameter_count():
    config = ModelConfig(d_model=8, d_state=4, n_layers=2, expand=1,
                         dt_min=1e-3, dt_max=1e-1,
                         alpha_max=1.38629436112, dropout=0.0)
    counts = []
    for _variant in GC_MATRIX_VARIANTS:
        model = TemporalMambaModel(input_dim=7, signal_dim=6, num_outputs=3,
                                   model_config=config, generalized_coordinates=True)
        counts.append(sum(p.numel() for p in model.parameters()))
    assert len(set(counts)) == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_temporal_model.py -q`

Expected: constructor/forward reject the new arguments.

- [ ] **Step 3: Add the optional model path**

When `generalized_coordinates=True`, instantiate `GeneralizedCoordinatePredictor` and build every Mamba layer with `error_dim=3 * signal_dim`; do not instantiate or call the legacy `NextStepPredictor` for GC variants. First pass remains unchanged. For GC variants require coordinate targets/mask, align errors, apply the order mask, apply shuffled/noise replacement only to the injected tensor, and run the shared second pass. `vanilla` and `two_pass` in a GC matrix still instantiate the same modules but inject no error.

Keep every existing default and call site valid when `generalized_coordinates=False`.

- [ ] **Step 4: Write failing masked-loss tests**

Add:

```python
def test_gc_auxiliary_loss_ignores_invalid_and_inactive_orders():
    errors = torch.zeros(2, 5, 12)
    errors[:, :, 4:] = 1000
    valid = torch.ones(2, 5, 3, 1, dtype=torch.bool)
    loss = generalized_prediction_loss(errors, valid, signal_dim=4, active_order=1)
    assert loss == 0


def test_gc_order_weights_are_preregistered():
    errors = torch.ones(1, 4, 6)
    valid = torch.ones(1, 4, 3, 1, dtype=torch.bool)
    value = generalized_prediction_loss(errors, valid, signal_dim=2, active_order=3)
    torch.testing.assert_close(value, torch.tensor(0.875))
```

Use Smooth L1 and fixed order weights `(1.0, 0.5, 0.25)` so the all-ones result is `0.875`.

- [ ] **Step 5: Implement and route the loss**

Implement `generalized_prediction_loss` with Smooth L1 against zero per active order and valid mask. In `compute_task_loss`, treat both `uci_har` and `generalized_dynamics` as multiclass cross-entropy tasks. In `compute_total_loss`, GC variants use the existing warmup and `lambda_aux`, while legacy variants retain their current branch exactly. Extend `multiclass_metrics` with `"balanced_accuracy": float(sum(recalls) / num_classes)` without changing any existing metric values.

- [ ] **Step 6: Run model/loss regressions**

Run:

```powershell
pytest tests/test_temporal_model.py tests/test_temporal_losses_metrics.py tests/test_query_binding.py tests/test_direct_scan.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add temporal_mamba/model.py temporal_mamba/losses.py temporal_mamba/metrics.py tests/test_temporal_model.py tests/test_temporal_losses_metrics.py
git commit -m "feat: integrate generalized-coordinate error path"
```

---

### Task 6: Add UCI HAR coordinate targets and OOD views

**Files:**
- Modify: `temporal_mamba/datasets/uci_har.py`
- Modify: `tests/test_uci_har.py`

**Interfaces:**
- `UCIHARDataset(..., transform)` accepts `none`, `reverse`, `shuffle`, `prefix50`, `noise_025`.
- GC-ready items include `coordinate_targets` and `coordinate_mask`; existing keys stay unchanged.

- [ ] **Step 1: Write failing causality and deterministic-view tests**

Add:

```python
def test_gc_targets_and_ood_views_are_causal_and_deterministic(tmp_path):
    root = tmp_path / "har"
    make_extracted_fixture(root)
    prepare_uci_har(root, data_seed=20260716)
    base = UCIHARDataset(root, "test", transform="none")[0]
    prefix = UCIHARDataset(root, "test", transform="prefix50")[0]
    noise_a = UCIHARDataset(root, "test", transform="noise_025")[0]
    noise_b = UCIHARDataset(root, "test", transform="noise_025")[0]
    assert prefix["signal"].shape == (64, 9)
    assert prefix["coordinate_targets"].shape == (64, 3, 9)
    np.testing.assert_array_equal(noise_a["signal"], noise_b["signal"])
    assert not np.array_equal(noise_a["signal"], base["signal"])
    assert noise_a["target"] == base["target"]
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_uci_har.py -q`

Expected: `prefix50` is rejected.

- [ ] **Step 3: Implement transforms and coordinates**

`prefix50` returns the first 64 samples. `noise_025` derives a sample seed from SHA-256 of `data_seed:sample_id:noise_025`, adds Gaussian noise with standard deviation `0.25` in normalized signal units, and retains the label. Compute backward-difference coordinates after the view transform and return float32 arrays plus the validity mask.

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_uci_har.py tests/test_temporal_training.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add temporal_mamba/datasets/uci_har.py tests/test_uci_har.py
git commit -m "feat: add UCI HAR generalized-coordinate views"
```

---

### Task 7: Route datasets, coordinate tensors, views and artifacts through training

**Files:**
- Modify: `temporal_mamba/train.py`
- Create: `configs/generalized_dynamics_gc.json`
- Create: `configs/uci_har_gc.json`
- Modify: `tests/test_temporal_training.py`

**Interfaces:**
- `BatchTensors` adds optional `coordinate_targets` and `coordinate_mask`.
- GC runs write artifact schema version 3 and metrics for exact preregistered views.
- `generalized_dynamics` outputs 3 classes; UCI HAR outputs 6 classes.

- [ ] **Step 1: Write failing batch and view tests**

Add tests that `_move_batch` preserves coordinate tensors; `build_datasets` returns seven synthetic splits; UCI GC evaluation exposes `val`, `test`, `prefix50`, `noise_025`; and legacy v2 still exposes its existing six views.

Use exact assertions:

```python
assert moved.coordinate_targets.shape == (2, 16, 3, 6)
assert moved.coordinate_mask.shape == (2, 16, 3, 1)
assert set(gc_datasets) == {
    "train", "val", "test", "length_256", "length_512", "parameter_ood", "noise_ood"
}
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_temporal_training.py -q`

Expected: `BatchTensors` and `build_datasets` lack the new contract.

- [ ] **Step 3: Implement dataset routing**

For `generalized_dynamics`, build/load the immutable manifest and return every split. For `config.uses_gc` UCI runs, retain standard train/val/test and add evaluation-only `prefix50` and `noise_025`. For legacy UCI runs, keep existing behavior. Instantiate `TemporalMambaModel(generalized_coordinates=config.uses_gc)` so GC-matrix `vanilla` and `two_pass` contain the same dormant K=3 modules and parameter count as K1/K2/K3.

Map synthetic sizes exactly as follows: `train_size → train`, `val_size → val`, `test_size → test`, and `long_test_size → each of length_256, length_512, parameter_ood, noise_ood`. Update `_classification_metrics` to accept `num_classes`; pass `config.num_outputs` from evaluation so synthetic metrics use three classes and UCI HAR uses six.

- [ ] **Step 4: Pass deterministic control seeds**

During training use `error_control_seed = config.seed * 1_000_003 + global_step`. During evaluation use `config.seed * 1_000_003 + batch_index`. Record the formula and actual seed in diagnostics so replay is exact.

- [ ] **Step 5: Extend guards and artifact schema**

Guard coordinate errors/masks for shapes and finite values. Schema v3 final artifacts record `gc_order`, per-order auxiliary losses, per-order error RMS, `dt` diagnostics, parameter count, dataset/view metrics, Git/config/manifest hashes and environment. Legacy runs retain their existing schema versions.

- [ ] **Step 6: Add exact configurations**

`configs/generalized_dynamics_gc.json`:

```json
{
  "dataset": "generalized_dynamics",
  "generalized_coordinates": true,
  "input_mode": "standard",
  "data_seed": 20260717,
  "signal_dim": 6,
  "num_outputs": 3,
  "seq_len": 128,
  "data": {"train_size": 18000, "val_size": 3000, "test_size": 3000, "long_test_size": 3000, "validation_fraction": 0.0},
  "model": {"d_model": 64, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
  "training": {"epochs": 40, "batch_size": 128, "lr": 0.001, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 10}
}
```

`configs/uci_har_gc.json` is exact and GC behavior comes from the runtime variant:

```json
{
  "dataset": "uci_har",
  "generalized_coordinates": true,
  "data_seed": 20260716,
  "signal_dim": 9,
  "num_outputs": 6,
  "seq_len": 128,
  "data": {"train_size": 0, "val_size": 0, "test_size": 0, "long_test_size": 0, "validation_fraction": 0.2},
  "model": {"d_model": 96, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
  "training": {"epochs": 40, "batch_size": 64, "lr": 0.0005, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 10}
}
```

- [ ] **Step 7: Run training-contract regressions**

Run:

```powershell
pytest tests/test_temporal_training.py tests/test_temporal_checkpoint_numerics.py tests/test_temporal_logic_v2.py -q
```

Expected: all tests pass and legacy artifact assertions remain unchanged.

- [ ] **Step 8: Commit**

```powershell
git add temporal_mamba/train.py configs/generalized_dynamics_gc.json configs/uci_har_gc.json tests/test_temporal_training.py
git commit -m "feat: route generalized-coordinate training artifacts"
```

---

### Task 8: Add the dedicated matrix runner and preregistered summarizer

**Files:**
- Create: `temporal_mamba/run_gc_matrix.py`
- Create: `temporal_mamba/summarize_gc.py`
- Create: `tests/test_gc_matrix_summary.py`

**Interfaces:**
- `expand_gc_matrix(stage: Literal["smoke", "screen", "confirm"]) -> tuple[RunSpec, ...]`.
- `summarize_gc_matrix(artifact_root, stage) -> dict[str, object]`.
- Final decision is exactly `supported`, `uncertain`, or `not_supported`.

- [ ] **Step 1: Write failing matrix-expansion tests**

```python
def test_screen_and_confirm_matrix_sizes():
    smoke = expand_gc_matrix("smoke")
    screen = expand_gc_matrix("screen")
    confirm = expand_gc_matrix("confirm")
    assert len(smoke) == 2 * 7
    assert len(screen) == 2 * 7 * 3
    assert len(confirm) == 2 * 7 * 5
    assert {run.seed for run in smoke} == {42}
    assert {run.seed for run in screen} == {42, 123, 777}
    assert {run.seed for run in confirm} == {42, 123, 777, 2026, 31415}
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_gc_matrix_summary.py -q`

Expected: module import fails.

- [ ] **Step 3: Implement the isolated runner**

Reuse `RunSpec` and the safe subprocess/status/reuse patterns from `run_matrix.py`, but hard-code the two GC configs and `GC_MATRIX_VARIANTS`. `smoke` expands one seed and passes the training CLI's epoch/tiny-batch overrides; `screen` expands the three screening seeds; `confirm` expands all five seeds. Require stage choice; never default to the legacy matrix. Reject completed artifacts unless run ID, Git commit, config hash and manifest hash match.

- [ ] **Step 4: Write failing statistical decision tests**

Generate fixture finals for all jobs and assert:

```python
assert summarize_gc_matrix(supported_root, "confirm")["decision"] == "supported"
assert summarize_gc_matrix(one_domain_root, "confirm")["decision"] == "uncertain"
assert summarize_gc_matrix(control_tie_root, "confirm")["decision"] == "not_supported"
```

Also assert mixed commits, missing jobs, non-finite metrics, invalid `dt`, and wrong manifests raise `ValueError`.

- [ ] **Step 5: Implement paired statistics and decision**

Use paired per-seed differences. For five seeds, calculate sample standard deviation and `mean ± 2.776445105 * s / sqrt(5)`. Synthetic primary OOD is mean balanced accuracy across `length_512`, `parameter_ood`, `noise_ood`. UCI primary OOD is mean macro-F1 across `prefix50`, `noise_025`. Apply the approved 2pp/1pp thresholds, lower-CI rule, control comparisons and ≤1pp ID degradation rule exactly.

- [ ] **Step 6: Emit machine-readable and Chinese reports**

Write `summary.json` atomically and `report_zh.md` containing job completeness, provenance, per-seed values, paired effects, confidence intervals, controls, ID/OOD metrics, and one unambiguous decision. Do not select best seeds.

- [ ] **Step 7: Run tests and commit**

Run:

```powershell
pytest tests/test_gc_matrix_summary.py tests/test_temporal_matrix_summary.py tests/test_v2_gate_summary.py -q
```

Expected: all tests pass.

Commit:

```powershell
git add temporal_mamba/run_gc_matrix.py temporal_mamba/summarize_gc.py tests/test_gc_matrix_summary.py
git commit -m "feat: add generalized-coordinate matrix decision gate"
```

---

### Task 9: Run the full regression suite and request code review

**Files:**
- Review all files changed in Tasks 2-8.

**Interfaces:**
- Produces: implementation candidate safe to sync to GitHub and Pro 6000.

- [ ] **Step 1: Run static compilation**

Run:

```powershell
python -m compileall -q temporal_mamba tests
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run the complete local unit suite**

Run:

```powershell
pytest -q
```

Expected: all tests pass; these are engineering checks, not scientific results.

- [ ] **Step 3: Verify legacy contracts explicitly**

Run:

```powershell
pytest tests/test_temporal_logic.py tests/test_temporal_logic_v2.py tests/test_query_binding.py tests/test_uci_har.py tests/test_direct_scan.py -q
```

Expected: all pass.

- [ ] **Step 4: Inspect the diff and provenance**

Run:

```powershell
git diff checkpoint/query-mamba-v2-before-gc-v1 --check
git status --short --branch
git log --oneline checkpoint/query-mamba-v2-before-gc-v1..HEAD
```

Expected: no whitespace errors, only scoped changes, and a clean feature branch.

- [ ] **Step 5: Use the requesting-code-review skill**

Review against the approved design, with special attention to causal leakage, parameter matching, legacy behavior, deterministic controls, artifact mixing and decision thresholds. Fix every confirmed issue with a failing regression test and a focused commit.

---

### Task 10: Sync the exact commit and execute Stage 1 on Pro 6000

**Files:**
- No new source files unless a Pro-only defect is reproduced and fixed test-first.

**Interfaces:**
- Consumes: reviewed, clean feature-branch commit pushed to GitHub.
- Produces: authoritative Pro 6000 environment record and Stage 1 artifact gate.

- [ ] **Step 1: Push branch and baseline tag**

Run only after network approval:

```powershell
git push -u origin exp/generalized-coordinate-minimal-ablation
git push origin checkpoint/query-mamba-v2-before-gc-v1
```

Expected: GitHub reports both refs updated.

- [ ] **Step 2: Sync the Pro 6000 checkout to the exact branch commit**

Run over SSH:

```bash
cd /home/lab/pcs3-phase1a
git fetch origin exp/generalized-coordinate-minimal-ablation --tags
git switch exp/generalized-coordinate-minimal-ablation
git pull --ff-only origin exp/generalized-coordinate-minimal-ablation
git rev-parse HEAD
```

Expected: remote HEAD equals the reviewed local commit. If the remote checkout is dirty, stop and preserve it; do not reset or overwrite it.

- [ ] **Step 3: Record hardware and software**

Run:

```bash
nvidia-smi
nproc --all
python -c "import json,platform,torch; print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"
```

Expected: NVIDIA Pro 6000 is visible and PyTorch CUDA is available.

- [ ] **Step 4: Run the authoritative regression suite with CPUs 8/9 excluded**

Run:

```bash
taskset --cpu-list 0-7,10-31 pytest -q
```

Expected: all tests pass. If `nproc --all` is not 32, construct the explicit list of every available CPU except 8 and 9, record it, and use that same list for every subsequent command.

- [ ] **Step 5: Generate datasets and verify manifests**

Run:

```bash
taskset --cpu-list 0-7,10-31 python -m temporal_mamba.datasets.generalized_dynamics --root /home/lab/datasets/pcs3-generalized-coordinate-v1/generalized_dynamics --config configs/generalized_dynamics_gc.json
taskset --cpu-list 0-7,10-31 python -m temporal_mamba.datasets.uci_har --root /home/lab/datasets/pcs3-generalized-coordinate-v1/uci_har --data-seed 20260716
```

Expected: both commands print finite manifests with hashes; rerunning leaves hashes unchanged.

- [ ] **Step 6: Run Stage 1 smoke for every variant on both datasets**

Run the dedicated runner in smoke mode:

```bash
taskset --cpu-list 0-7,10-31 python -m temporal_mamba.run_gc_matrix --stage smoke --data-root /home/lab/datasets/pcs3-generalized-coordinate-v1 --artifact-root artifacts/gc-v1-smoke --device cuda
```

Expected: 14/14 smoke jobs complete; tiny-batch gates pass; all tensors are finite; `dt` remains within `[0.001, 0.1]`; parameter counts are identical.

- [ ] **Step 7: Commit a Pro-only fix only if a deterministic failing test requires it**

Any fix follows test-fail → minimal fix → full local tests → push → ff-only remote sync → full Pro tests. Do not patch the remote checkout directly.

---

### Task 11: Execute screening, confirmation and checkpoint decision on Pro 6000

**Files:**
- Generated remotely: `artifacts/gc-v1-screen/**`
- Generated remotely: `artifacts/gc-v1-confirm/**`
- Generated remotely: `artifacts/gc-v1-confirm/summary.json`
- Generated remotely: `artifacts/gc-v1-confirm/report_zh.md`

**Interfaces:**
- Produces: preregistered decision and either a continued GC branch or a clean query-conditioned rollback point.

- [ ] **Step 1: Launch the 42-job screen matrix**

Run:

```bash
taskset --cpu-list 0-7,10-31 python -m temporal_mamba.run_gc_matrix --stage screen --data-root /home/lab/datasets/pcs3-generalized-coordinate-v1 --artifact-root artifacts/gc-v1-screen --device cuda
```

Expected: exactly 42 complete jobs: 2 datasets × 7 variants × 3 seeds.

- [ ] **Step 2: Validate and summarize screening**

Run:

```bash
python -m temporal_mamba.summarize_gc --stage screen --artifact-root artifacts/gc-v1-screen
```

Expected: strict validation passes and report states whether neither or at least one domain passed the fixed screening rule.

- [ ] **Step 3: Stop or launch confirmation according to the gate**

If neither domain passes, do not run confirmation. Preserve artifacts and proceed to Step 6. If at least one passes, run:

```bash
taskset --cpu-list 0-7,10-31 python -m temporal_mamba.run_gc_matrix --stage confirm --data-root /home/lab/datasets/pcs3-generalized-coordinate-v1 --artifact-root artifacts/gc-v1-confirm --device cuda
```

Expected: exactly 70 complete jobs: 2 datasets × 7 variants × 5 seeds. Reuse of the original three seeds is allowed only when Git/config/manifest hashes match exactly.

- [ ] **Step 4: Produce the final preregistered decision**

Run:

```bash
python -m temporal_mamba.summarize_gc --stage confirm --artifact-root artifacts/gc-v1-confirm
```

Expected: strict validation passes and `decision` is exactly one of `supported`, `uncertain`, `not_supported`.

- [ ] **Step 5: Verify result completeness before any scientific claim**

Run:

```bash
python -c "import json; p=json.load(open('artifacts/gc-v1-confirm/summary.json')); print(p['decision'], p['completed_jobs'], p['git_commit'], p['dataset_manifest_hashes'])"
```

Expected: decision, 70 jobs, one Git commit, and one manifest hash per dataset. For a screen stop, inspect the analogous screen summary and require 42 jobs.

- [ ] **Step 6: Apply the checkpoint decision**

If `supported`, retain the feature branch for the separately designed Predictor-position experiment. If `uncertain`, freeze claims and request a new preregistration before adding seeds. If `not_supported` or screening stopped, create and push the termination tag on the final experimental commit:

```bash
git tag -a checkpoint/gc-v1-terminated -m "Generalized-coordinate minimal ablation did not pass preregistered gate"
git push origin checkpoint/gc-v1-terminated
```

Then continue query-conditioned Temporal Mamba from `checkpoint/query-mamba-v2-before-gc-v1` on a new, separately designed branch; do not make GC the default path.

- [ ] **Step 7: Copy the Chinese report into the repository without copying large artifacts**

Add only the final Chinese Markdown report and a compact machine-readable summary under `docs/`; reference remote artifact roots and hashes. Run `git diff --check`, commit the report, and push the feature branch. Do not commit checkpoints, datasets, NPZ files, or full training logs.
