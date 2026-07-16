# Temporal Mamba Causal Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, deploy, and execute a stable `B x T x D` Temporal Mamba experiment with synthetic temporal logic, raw UCI HAR signals, six causal variants, and three seeds per variant.

**Architecture:** Add an isolated `temporal_mamba` package without modifying the historical image experiment. A directly recurrent selective SSM supplies the sequence backbone; a detached first pass predicts next-step position and velocity errors, and a bounded error pathway modulates `dt` during a shared-weight second pass. Dataset backends expose the same dictionary batch contract, allowing one training engine and one 36-run matrix.

**Tech Stack:** Python 3.10+, PyTorch 2.x, NumPy, standard library `urllib`/`zipfile`, pytest; NVIDIA CUDA on `lab@100.68.12.20` for integration and full runs.

## Global Constraints

- Preserve GitHub `master` history and all existing K=1/K=2/K=3 image code; new code lives under `temporal_mamba/`.
- Do not copy the stale Pro 6000 image files over newer GitHub files.
- Authoritative signal contract is `B x T x D`, with time at dimension 1.
- Direct recurrence and exponentials run in float32.
- `dt_min=1e-3`, `dt_max=1e-1`, `alpha_max=log(4)`, and error modulation starts at exactly zero.
- Dataset manifests use fixed `data_seed=20260716`; training seeds are exactly `42`, `123`, and `777`.
- Full matrix is six variants on two datasets and three seeds: exactly 36 runs.
- Any non-finite loss, gradient, parameter, activation, or `dt` aborts the run and writes a failure artifact.
- Full runs may start only after unit tests, tiny-batch overfit gates, and twelve one-epoch smoke runs pass.

---

### Task 1: Package Contract and Configuration

**Files:**
- Create: `temporal_mamba/__init__.py`
- Create: `temporal_mamba/config.py`
- Create: `configs/temporal_logic.json`
- Create: `configs/uci_har.json`
- Test: `tests/test_temporal_config.py`

**Interfaces:**
- Produces: `VARIANTS`, `TRAINING_SEEDS`, `ModelConfig`, `TrainingConfig`, `DataConfig`, `ExperimentConfig`, `load_experiment_config(path, variant, seed)`.
- `ExperimentConfig` exposes `pass_count`, `uses_error`, `uses_aux`, and `time_transform` as derived properties.

- [ ] **Step 1: Write the failing configuration tests**

```python
from temporal_mamba.config import TRAINING_SEEDS, VARIANTS, load_experiment_config


def test_ablation_contract_is_exact():
    assert VARIANTS == (
        "vanilla", "two_pass", "error_inject", "error_aux",
        "time_shuffle", "time_reverse",
    )
    assert TRAINING_SEEDS == (42, 123, 777)


def test_variant_properties(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text('''{
      "dataset":"temporal_logic","data_seed":20260716,"signal_dim":8,
      "num_outputs":1,"seq_len":128,
      "data":{"train_size":120,"val_size":60,"test_size":60,
              "long_test_size":60,"validation_fraction":0.0},
      "model":{"d_model":64,"d_state":16,"n_layers":4,"expand":2,
               "dt_min":0.001,"dt_max":0.1,"alpha_max":1.38629436112,
               "dropout":0.1},
      "training":{"epochs":30,"batch_size":128,"lr":0.001,
                  "weight_decay":0.01,"warmup_fraction":0.05,
                  "lambda_aux":0.1,"aux_warmup_fraction":0.1,"patience":8}
    }''')
    cfg = load_experiment_config(path, variant="time_reverse", seed=42)
    assert cfg.pass_count == 2
    assert cfg.uses_error and cfg.uses_aux
    assert cfg.time_transform == "reverse"
    assert cfg.model.dt_min < cfg.model.dt_max
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python -m pytest tests/test_temporal_config.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'temporal_mamba'`.

- [ ] **Step 3: Implement frozen dataclasses and strict validation**

```python
VARIANTS = ("vanilla", "two_pass", "error_inject", "error_aux", "time_shuffle", "time_reverse")
TRAINING_SEEDS = (42, 123, 777)

@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    data_seed: int
    signal_dim: int
    num_outputs: int
    seq_len: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    variant: str
    seed: int

    @property
    def pass_count(self) -> int:
        return 1 if self.variant == "vanilla" else 2

    @property
    def uses_error(self) -> bool:
        return self.variant in {"error_inject", "error_aux", "time_shuffle", "time_reverse"}

    @property
    def uses_aux(self) -> bool:
        return self.variant in {"error_aux", "time_shuffle", "time_reverse"}

    @property
    def time_transform(self) -> str:
        return {"time_shuffle": "shuffle", "time_reverse": "reverse"}.get(self.variant, "none")
```

`load_experiment_config` rejects unknown keys, unknown datasets/variants, non-approved seeds, non-positive dimensions, invalid fractions, and `dt_min >= dt_max` with `ValueError` messages naming the field.

- [ ] **Step 4: Add fixed dataset-level JSON configurations**

`configs/temporal_logic.json` contains the following fixed data and optimization values in addition to the model object shown in Step 1:

```json
{
  "dataset": "temporal_logic",
  "data_seed": 20260716,
  "signal_dim": 8,
  "num_outputs": 1,
  "seq_len": 128,
  "data": {"train_size": 12000, "val_size": 2400, "test_size": 2400, "long_test_size": 2400, "validation_fraction": 0.0},
  "model": {"d_model": 64, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
  "training": {"epochs": 30, "batch_size": 128, "lr": 0.001, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 8}
}
```

`configs/uci_har.json` contains:

```json
{
  "dataset": "uci_har",
  "data_seed": 20260716,
  "signal_dim": 9,
  "num_outputs": 6,
  "seq_len": 128,
  "data": {"train_size": 0, "val_size": 0, "test_size": 0, "long_test_size": 0, "validation_fraction": 0.2},
  "model": {"d_model": 96, "d_state": 16, "n_layers": 4, "expand": 2, "dt_min": 0.001, "dt_max": 0.1, "alpha_max": 1.38629436112, "dropout": 0.1},
  "training": {"epochs": 40, "batch_size": 64, "lr": 0.0005, "weight_decay": 0.01, "warmup_fraction": 0.05, "lambda_aux": 0.1, "aux_warmup_fraction": 0.1, "patience": 10}
}
```

- [ ] **Step 5: Run the configuration tests**

Run: `python -m pytest tests/test_temporal_config.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add temporal_mamba/__init__.py temporal_mamba/config.py configs tests/test_temporal_config.py
git commit -m "feat: define temporal experiment contracts"
```

---

### Task 2: Temporal Logic Evaluator and Manifest Builder

**Files:**
- Create: `temporal_mamba/datasets/__init__.py`
- Create: `temporal_mamba/datasets/temporal_logic.py`
- Test: `tests/test_temporal_logic.py`

**Interfaces:**
- Produces: `TemporalQuery`, `evaluate_query(signal, query)`, `encode_query(query, event_dim, seq_len)`, `build_temporal_logic_manifest(root, sizes, data_seed)`, and `TemporalLogicDataset(root, split, transform)`.
- Dataset items are dictionaries with `features`, `signal`, `target`, `sample_id`, `formula_family`, and `base_target`.

- [ ] **Step 1: Write hand-checked evaluator tests**

```python
import numpy as np
from temporal_mamba.datasets.temporal_logic import TemporalQuery, evaluate_query


def test_before_and_reversal_change_truth():
    x = np.zeros((8, 3), dtype=np.float32)
    x[1, 0] = 1
    x[6, 1] = 1
    q = TemporalQuery("BEFORE", event_a=0, event_b=1)
    assert evaluate_query(x, q) is True
    assert evaluate_query(x[::-1].copy(), q) is False


def test_bounded_response_checks_every_trigger():
    x = np.zeros((10, 2), dtype=np.float32)
    x[[1, 6], 0] = 1
    x[[3, 9], 1] = 1
    assert evaluate_query(x, TemporalQuery("BOUNDED_RESPONSE", 0, 1, p0=3))
    x[9, 1] = 0
    assert not evaluate_query(x, TemporalQuery("BOUNDED_RESPONSE", 0, 1, p0=3))
```

Add equivalent positive/negative cases for `EVENTUALLY`, `UNTIL`, `COUNT_WITHIN`, and `GAP`.

- [ ] **Step 2: Run evaluator tests and verify failure**

Run: `python -m pytest tests/test_temporal_logic.py -q`

Expected: FAIL because the dataset module does not exist.

- [ ] **Step 3: Implement formula dataclass, evaluator, and query encoding**

```python
@dataclass(frozen=True)
class TemporalQuery:
    family: str
    event_a: int
    event_b: int = -1
    p0: int = 0
    p1: int = 0
    p2: int = 0

def encode_query(query: TemporalQuery, event_dim: int, seq_len: int) -> np.ndarray:
    family = np.zeros(len(FORMULA_FAMILIES), np.float32)
    family[FORMULA_FAMILIES.index(query.family)] = 1
    a = np.eye(event_dim, dtype=np.float32)[query.event_a]
    b = np.zeros(event_dim, np.float32) if query.event_b < 0 else np.eye(event_dim, dtype=np.float32)[query.event_b]
    bounds = np.asarray([query.p0, query.p1, query.p2], np.float32) / max(seq_len - 1, 1)
    return np.concatenate([family, a, b, bounds])
```

The evaluator uses explicit NumPy operations and returns a Python `bool`. `UNTIL(A,B)` requires B and requires A at every index before the first B. `BOUNDED_RESPONSE` checks every A trigger.

- [ ] **Step 4: Write failing generator and split tests**

```python
def test_manifest_is_balanced_verified_and_reproducible(tmp_path):
    sizes = {"train": 120, "val": 60, "test": 60, "long_test": 60}
    m1 = build_temporal_logic_manifest(tmp_path / "a", sizes, data_seed=20260716)
    m2 = build_temporal_logic_manifest(tmp_path / "b", sizes, data_seed=20260716)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    for split in sizes:
        data = np.load(tmp_path / "a" / f"{split}.npz", allow_pickle=False)
        assert len(data["target"]) == sizes[split]
        for family in range(6):
            labels = data["target"][data["family"] == family]
            assert abs(int(labels.sum()) - (len(labels) - int(labels.sum()))) <= 1
    assert m1["cross_split_duplicates"] == 0
```

- [ ] **Step 5: Implement constructive generation and immutable NPZ manifests**

Use one constructor per formula family and label. Add unrelated background events only after the required relevant-channel pattern is fixed. Verify every generated sample by calling `evaluate_query`; retry with a deterministic child RNG if verification fails. Save arrays, split child seeds, counts, duplicate check, and SHA-256 values in `manifest.json` using atomic replacement.

- [ ] **Step 6: Implement deterministic reverse/shuffle dataset views**

`TemporalLogicDataset.__getitem__` reverses or permutes only `signal`, recomputes `target`, retains `base_target`, rebuilds normalized time, broadcasts encoded query, and concatenates `[signal, time, query]` into `features`. Shuffle RNG is derived from `data_seed` and `sample_id`, never the training seed.

- [ ] **Step 7: Run temporal-logic tests**

Run: `python -m pytest tests/test_temporal_logic.py -q`

Expected: evaluator, generation, balance, duplicate, reproducibility, and transform tests all pass.

- [ ] **Step 8: Commit**

```bash
git add temporal_mamba/datasets tests/test_temporal_logic.py
git commit -m "feat: add verified temporal logic benchmark"
```

---

### Task 3: Direct Selective SSM and Bounded dt

**Files:**
- Create: `temporal_mamba/ssm.py`
- Test: `tests/test_direct_scan.py`

**Interfaces:**
- Produces: `inverse_softplus`, `direct_selective_scan`, `DirectSelectiveSSM`, `TemporalMambaBlock`, and `SSMDiagnostics`.
- `DirectSelectiveSSM.forward(u, error=None, return_diagnostics=False)` consumes `B x T x D_inner` and optional `B x T x D_error`.

- [ ] **Step 1: Write independent forward and gradient equivalence tests**

```python
def reference_scan(u, dt, a_log, b, c, d_skip):
    a = -torch.exp(a_log.float())
    h = torch.zeros(u.size(0), u.size(2), a.size(1), dtype=torch.float32)
    ys = []
    for t in range(u.size(1)):
        a_bar = torch.exp(dt[:, t].float().unsqueeze(-1) * a)
        b_bar = dt[:, t].float().unsqueeze(-1) * b[:, t].float().unsqueeze(1)
        h = a_bar * h + b_bar * u[:, t].float().unsqueeze(-1)
        ys.append((h * c[:, t].float().unsqueeze(1)).sum(-1) + d_skip * u[:, t].float())
    return torch.stack(ys, 1)

def test_direct_scan_matches_reference_and_gradients():
    # Use separate cloned tensors for implementation and reference.
    # Assert max forward difference < 1e-6 and every input gradient difference < 1e-5.
```

- [ ] **Step 2: Run the scan tests and verify failure**

Run: `python -m pytest tests/test_direct_scan.py -q`

Expected: FAIL because `temporal_mamba.ssm` does not exist.

- [ ] **Step 3: Implement exact float32 recurrence**

```python
def direct_selective_scan(u, dt, a_log, b, c, d_skip):
    u32, dt32 = u.float(), dt.float()
    a = -torch.exp(a_log.float())
    h = torch.zeros(u.size(0), u.size(2), a.size(1), device=u.device, dtype=torch.float32)
    ys = []
    for t in range(u.size(1)):
        a_bar = torch.exp(dt32[:, t].unsqueeze(-1) * a)
        b_bar = dt32[:, t].unsqueeze(-1) * b[:, t].float().unsqueeze(1)
        h = a_bar * h + b_bar * u32[:, t].unsqueeze(-1)
        ys.append((h * c[:, t].float().unsqueeze(1)).sum(-1) + d_skip.float() * u32[:, t])
    return torch.stack(ys, dim=1)
```

- [ ] **Step 4: Write initialization and extreme-error tests**

```python
def test_zero_error_scale_matches_base_dt_and_extremes_are_bounded():
    layer = DirectSelectiveSSM(16, d_state=4, error_dim=6, dt_min=1e-3, dt_max=1e-1)
    u = torch.randn(2, 32, 16)
    zero = layer.compute_dt(u, error=None)
    assert torch.equal(zero, layer.compute_dt(u, error=torch.randn(2, 32, 6) * 1e8))
    layer.alpha_raw.data.fill_(100)
    bounded = layer.compute_dt(u, error=torch.randn(2, 32, 6) * 1e8)
    assert bounded.min() >= 1e-3 and bounded.max() <= 1e-1
    assert torch.equal(layer.a_log[0], torch.log(torch.arange(1, 5, dtype=torch.float32)))
```

- [ ] **Step 5: Implement official-style initialization and bounded modulation**

Initialize `a_log` with `log(1..N)`, `d_skip` with ones, `dt_rank=ceil(D/16)`, and the `dt_proj` bias with inverse softplus of log-uniform initial `dt`. Use `RMSNorm(error_dim)`, a bias-free error projection, and `alpha_max * tanh(alpha_raw)`. Clamp both base and modulated `dt` to configured bounds.

- [ ] **Step 6: Implement block gating and diagnostics**

`TemporalMambaBlock` applies pre-norm, value/gate projection, causal depthwise convolution, direct SSM, SiLU gate, output projection, dropout, and residual. `SSMDiagnostics` contains scalar tensors for `dt_min`, `dt_max`, `error_rms`, `error_max`, `output_rms`, `output_max`, and `finite`.

- [ ] **Step 7: Run the scan test suite**

Run: `python -m pytest tests/test_direct_scan.py -q`

Expected: forward/gradient equivalence, initialization, dt-bound, and diagnostic tests pass in float32 and float64 inputs.

- [ ] **Step 8: Commit**

```bash
git add temporal_mamba/ssm.py tests/test_direct_scan.py
git commit -m "feat: add stable direct selective recurrence"
```

---

### Task 4: Temporal Two-Pass Model and Predictor Alignment

**Files:**
- Create: `temporal_mamba/model.py`
- Test: `tests/test_temporal_model.py`

**Interfaces:**
- Produces: `TemporalModelOutput`, `NextStepPredictor`, and `TemporalMambaModel(input_dim, signal_dim, num_outputs, model_config)`.
- `TemporalMambaModel.forward(features, signal, variant, return_diagnostics=False)` returns logits, aligned errors, pass count, and diagnostics.

- [ ] **Step 1: Write failing shape and pass-contract tests**

```python
@pytest.mark.parametrize("variant,passes,uses_error", [
    ("vanilla", 1, False), ("two_pass", 2, False),
    ("error_inject", 2, True), ("error_aux", 2, True),
    ("time_shuffle", 2, True), ("time_reverse", 2, True),
])
def test_variant_forward_contract(variant, passes, uses_error):
    model = make_tiny_model()
    features = torch.randn(3, 12, 20)
    signal = torch.randn(3, 12, 4)
    out = model(features, signal, variant=variant, return_diagnostics=True)
    assert out.logits.shape == (3, 1)
    assert out.pass_count == passes
    assert (out.position_error is not None) is uses_error
    assert (out.velocity_error is not None) is uses_error
```

- [ ] **Step 2: Run model tests and verify failure**

Run: `python -m pytest tests/test_temporal_model.py -q`

Expected: FAIL because model classes do not exist.

- [ ] **Step 3: Implement detached aligned next-step prediction**

```python
def aligned_errors(hidden, signal, predictor):
    pred_x, pred_v = predictor(hidden.detach())
    pos = torch.zeros_like(signal)
    vel = torch.zeros_like(signal)
    pos[:, 1:] = signal[:, 1:].detach() - pred_x[:, :-1]
    delta = signal[:, 1:].detach() - signal[:, :-1].detach()
    vel[:, 1:] = delta - pred_v[:, :-1]
    return pos, vel
```

Tests assert that targets and first-pass hidden receive no gradient through `aligned_errors`, predictor parameters do receive gradients, and index `t` compares predictions from `t-1` to signals at `t`.

- [ ] **Step 4: Implement shared-weight one/two-pass encoder flow**

`vanilla` classifies first-pass hidden. Every other variant runs the same encoder a second time from zero state. `two_pass` supplies `error=None`; the remaining four concatenate aligned position and velocity errors and supply them to every SSM block. `time_shuffle` and `time_reverse` differ only in dataset view, not model internals.

- [ ] **Step 5: Add causal final-step heads and diagnostics aggregation**

After output normalization, select `hidden[:, -1]`. Use one logit for temporal logic and six logits for HAR. Aggregate layer diagnostics with min/max semantics and include `pass_count`, `uses_error`, and finite status in `TemporalModelOutput`.

- [ ] **Step 6: Run model tests**

Run: `python -m pytest tests/test_temporal_model.py -q`

Expected: all shape, alignment, detach, variant, dt-bound, and finite-output tests pass for lengths 64, 128, and 256.

- [ ] **Step 7: Commit**

```bash
git add temporal_mamba/model.py tests/test_temporal_model.py
git commit -m "feat: add temporal two-pass error model"
```

---

### Task 5: UCI HAR Raw Signal Preparation

**Files:**
- Create: `temporal_mamba/datasets/uci_har.py`
- Test: `tests/test_uci_har.py`

**Interfaces:**
- Produces: `download_uci_har(root)`, `prepare_uci_har(root, data_seed)`, and `UCIHARDataset(root, split, transform)`.
- Dataset items match the temporal-logic dictionary contract, with empty query features and six-class integer targets.

- [ ] **Step 1: Write a miniature extracted-layout fixture and failing loader tests**

Create nine `Inertial Signals/*_train.txt` matrices, labels, and subject IDs under a pytest temporary directory. Assert output shape `B x 128 x 9`, fixed validation subjects, no subject overlap, train-only normalization, deterministic shuffle/reverse, and unchanged activity labels.

- [ ] **Step 2: Run HAR tests and verify failure**

Run: `python -m pytest tests/test_uci_har.py -q`

Expected: FAIL because the HAR module does not exist.

- [ ] **Step 3: Implement secure download and extraction**

Use `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`. Download to a temporary file, compute SHA-256, reject ZIP members whose resolved paths escape the extraction root, atomically move the verified archive into place, and record URL, SHA-256, size, and extraction members in `source_manifest.json`.

- [ ] **Step 4: Implement raw nine-channel stacking and subject split**

Load `body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, and `total_acc_{x,y,z}`. Stack on the last dimension. Select 20% of official training subjects with `np.random.default_rng(20260716)`, persist IDs, compute normalization from remaining training subjects only, and apply the same statistics to validation and official test subjects.

- [ ] **Step 5: Implement dataset views and run tests**

Shuffle permutations derive from `data_seed + sample_id`; reversal uses `signal[::-1]`. Add normalized time as a tenth model feature while preserving raw normalized sensor `signal` as nine channels for the predictor.

Run: `python -m pytest tests/test_uci_har.py -q`

Expected: all fixture, split, normalization, transform, and path-safety tests pass.

- [ ] **Step 6: Commit**

```bash
git add temporal_mamba/datasets/uci_har.py tests/test_uci_har.py
git commit -m "feat: add raw UCI HAR sequence loader"
```

---

### Task 6: Pointwise Auxiliary Loss and Metrics

**Files:**
- Create: `temporal_mamba/losses.py`
- Create: `temporal_mamba/metrics.py`
- Test: `tests/test_temporal_losses_metrics.py`

**Interfaces:**
- Produces: `auxiliary_weight`, `compute_task_loss`, `pointwise_prediction_loss`, `compute_total_loss`, `binary_metrics`, and `multiclass_metrics`.

- [ ] **Step 1: Write cancellation and warmup tests**

```python
def test_pointwise_loss_does_not_cancel_opposite_errors():
    pos = torch.tensor([[[0.0], [10.0], [-10.0]]])
    vel = torch.zeros_like(pos)
    loss = pointwise_prediction_loss(pos, vel, velocity_weight=0.5)
    assert loss > 8.0

def test_auxiliary_weight_warms_from_zero():
    assert auxiliary_weight(0, total_epochs=30, target=0.1, warmup_fraction=0.1) == 0.0
    assert auxiliary_weight(3, total_epochs=30, target=0.1, warmup_fraction=0.1) == pytest.approx(0.1)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_temporal_losses_metrics.py -q`

Expected: FAIL because loss and metric modules do not exist.

- [ ] **Step 3: Implement task and pointwise Smooth L1 losses**

Temporal logic uses `binary_cross_entropy_with_logits`; HAR uses cross entropy. Exclude index zero from prediction losses, apply `smooth_l1_loss` before reduction, and force auxiliary weight to zero for `vanilla`, `two_pass`, and `error_inject`.

- [ ] **Step 4: Implement dependency-free metrics**

Compute binary accuracy/balanced accuracy/F1 and multiclass accuracy/macro-F1/per-class recall/confusion matrix from integer arrays. Define zero-division values as zero and test missing-class cases.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_temporal_losses_metrics.py -q`

Expected: all loss, warmup, binary, and multiclass tests pass.

```bash
git add temporal_mamba/losses.py temporal_mamba/metrics.py tests/test_temporal_losses_metrics.py
git commit -m "feat: add temporal losses and metrics"
```

---

### Task 7: Atomic Checkpoints, RNG Restore, and Numerical Guards

**Files:**
- Create: `temporal_mamba/checkpoint.py`
- Create: `temporal_mamba/numerics.py`
- Test: `tests/test_temporal_checkpoint_numerics.py`

**Interfaces:**
- Produces: `save_checkpoint`, `load_checkpoint`, `NumericalFailure`, `assert_finite_tensor`, `assert_finite_model`, and `write_failure_artifact`.

- [ ] **Step 1: Write failing round-trip and failure-artifact tests**

The round-trip test performs one optimizer/scheduler step, saves all Python/NumPy/Torch RNG state, mutates everything, restores, and proves the next random values and next model update are identical. The failure test injects NaN into `dt` diagnostics and asserts `failure.json` contains run ID, batch, epoch, tensor name, and last healthy checkpoint.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_temporal_checkpoint_numerics.py -q`

Expected: FAIL because checkpoint/numerics modules do not exist.

- [ ] **Step 3: Implement atomic versioned checkpoints**

Write to `<path>.tmp`, flush and `os.fsync`, then `os.replace`. Store schema version, config hash, Git commit, dataset manifest hash, model/optimizer/scheduler/scaler states, RNG states, epoch/step, best metric, history cursor, and data-loader generator state.

- [ ] **Step 4: Implement explicit finite guards**

Guards call `torch.isfinite` on loss, gradients, parameters, logits, errors, SSM outputs, and dt diagnostics. `NumericalFailure` includes the failing component and observed min/max where finite. Failure artifacts are atomically written and the exception is re-raised.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_temporal_checkpoint_numerics.py -q`

Expected: round-trip and intentional-failure tests pass.

```bash
git add temporal_mamba/checkpoint.py temporal_mamba/numerics.py tests/test_temporal_checkpoint_numerics.py
git commit -m "feat: add reproducible checkpoints and guards"
```

---

### Task 8: Unified Training Engine and Tiny-Batch Gates

**Files:**
- Create: `temporal_mamba/train.py`
- Test: `tests/test_temporal_training.py`

**Interfaces:**
- Produces: `set_training_seed`, `build_datasets`, `build_loaders`, `train_epoch`, `evaluate`, `overfit_tiny_batch`, `run_training`, and CLI `python -m temporal_mamba.train`.

- [ ] **Step 1: Write failing one-step and resume integration tests**

Use an in-memory dictionary dataset. Assert one train step updates parameters, all six variants use expected pass/error/aux contracts, early-stopping selection uses validation only, final metrics are emitted, and resume produces the same next-step parameters as uninterrupted training.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_temporal_training.py -q`

Expected: FAIL because the training engine does not exist.

- [ ] **Step 3: Implement deterministic loaders and optimizer schedule**

Use seeded `torch.Generator`, a worker initializer derived from the training seed, AdamW, step-based 5% linear warmup, cosine decay, and `clip_grad_norm_(..., error_if_nonfinite=True)`. Do not enable AMP until all 36 float32 runs are complete.

- [ ] **Step 4: Implement epoch/evaluation loops and artifact schema**

Each run writes `config.json`, `environment.json`, `dataset_manifest.json`, append-only `history.jsonl`, `best.pt`, `last.pt`, and `final.json`. Evaluation includes original and transformed/frozen-label diagnostics required by the dataset. Only validation selects the best checkpoint.

- [ ] **Step 5: Implement tiny-batch overfit gates**

Temporal logic gate: 64 fixed samples, reach at least 98% training accuracy. HAR fixture gate: 48 fixed samples, reach at least 95%. Gates run with dropout disabled and no temporal transformation; failure prevents smoke/full execution.

- [ ] **Step 6: Run training tests and the complete unit suite**

Run: `python -m pytest tests/test_temporal_*.py -q`

Expected: all temporal tests pass.

- [ ] **Step 7: Commit**

```bash
git add temporal_mamba/train.py tests/test_temporal_training.py
git commit -m "feat: add guarded temporal training engine"
```

---

### Task 9: Matrix Runner and Completeness-Aware Summarizer

**Files:**
- Create: `temporal_mamba/run_matrix.py`
- Create: `temporal_mamba/summarize.py`
- Test: `tests/test_temporal_matrix_summary.py`

**Interfaces:**
- Produces: `RunSpec`, `expand_matrix`, `run_matrix`, `validate_matrix`, `summarize_matrix`, and both module CLIs.

- [ ] **Step 1: Write exact 36-run expansion test**

```python
def test_full_matrix_is_exact_and_unique():
    specs = expand_matrix(datasets=("temporal_logic", "uci_har"))
    assert len(specs) == 36
    assert len({s.run_id for s in specs}) == 36
    assert {s.seed for s in specs} == {42, 123, 777}
    assert {s.variant for s in specs} == set(VARIANTS)
```

- [ ] **Step 2: Write completeness and paired-delta tests**

Create 35 fake complete artifacts and assert `validate_matrix` names the missing run. Add the 36th and assert summary mean/sample-std plus paired deltas `two_pass-vanilla`, `error_inject-two_pass`, and `error_aux-error_inject` equal hand-calculated values.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_temporal_matrix_summary.py -q`

Expected: FAIL because matrix modules do not exist.

- [ ] **Step 4: Implement resumable sequential matrix execution**

The runner launches one subprocess at a time, records start/end/exit code, skips a run only if `final.json` has matching config hash, dataset hash, and Git commit, and otherwise resumes from `last.pt`. `--dry-run` prints all selected run IDs without mutation.

- [ ] **Step 5: Implement strict aggregation and Markdown report generation**

Reject missing, failed, non-finite, or metadata-mismatched runs. Write `summary.json`, `summary.csv`, and `docs/experiment_report.md` with dataset tables, per-family logic metrics, HAR confusion matrices, paired seed deltas, and shuffle/reversal diagnostics.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_temporal_matrix_summary.py -q`

Expected: expansion, completeness, resume-skip, aggregation, and report tests pass.

```bash
git add temporal_mamba/run_matrix.py temporal_mamba/summarize.py tests/test_temporal_matrix_summary.py
git commit -m "feat: add causal matrix runner and summary"
```

---

### Task 10: Remote Dependency and Dataset Preparation

**Files:**
- Modify: `requirements.txt`
- Create remotely/generated: `/home/lab/datasets/pcs3-temporal/uci_har/source_manifest.json`
- Create remotely/generated: `/home/lab/datasets/pcs3-temporal/temporal_logic/manifest.json`

**Interfaces:**
- Consumes all dataset preparation APIs.
- Produces immutable dataset manifests whose hashes enter every run.

- [ ] **Step 1: Add only missing runtime/test dependencies**

Retain existing requirements. Add version floors for `numpy`, `torch`, and `pytest` only when absent; do not add `mamba-ssm`, scikit-learn, pandas, or `ucimlrepo`.

- [ ] **Step 2: Run the full suite in the Pro 6000 environment**

Run remotely:

```bash
/home/lab/miniconda3/envs/evuav/bin/python -m pytest tests/test_temporal_*.py -q
```

Expected: all tests pass on CUDA-capable PyTorch 2.11.0+cu128.

- [ ] **Step 3: Prepare temporal logic and UCI HAR data**

Run remotely:

```bash
python -m temporal_mamba.datasets.temporal_logic --root /home/lab/datasets/pcs3-temporal/temporal_logic
python -m temporal_mamba.datasets.uci_har --root /home/lab/datasets/pcs3-temporal/uci_har
```

Expected: manifests exist, all listed hashes verify, UCI HAR shapes are `(7352,128,9)` train/validation source and `(2947,128,9)` official test before the fixed validation split.

- [ ] **Step 4: Commit dependency metadata**

```bash
git add requirements.txt
git commit -m "chore: declare temporal experiment dependencies"
```

---

### Task 11: Tiny-Batch and Twelve-Run Smoke Gate

**Files:**
- Generated: `artifacts/smoke/**`

- [ ] **Step 1: Run both tiny-batch gates on Pro 6000**

Run: `python -m temporal_mamba.train --config configs/temporal_logic.json --variant vanilla --seed 42 --overfit-only`

Run: `python -m temporal_mamba.train --config configs/uci_har.json --variant vanilla --seed 42 --overfit-only`

Expected: temporal logic >=98% and HAR >=95% on their fixed tiny batches, with finite diagnostics.

- [ ] **Step 2: Run all twelve dataset/variant one-epoch smoke runs**

Run: `python -m temporal_mamba.run_matrix --datasets temporal_logic uci_har --variants all --seeds 42 --epochs 1 --artifact-root artifacts/smoke`

Expected: exactly 12 `final.json` artifacts, zero failures, every variant's recorded pass/error/aux/transform fields match the table in the design spec.

- [ ] **Step 3: Validate smoke completeness and numerical bounds**

Run: `python -m temporal_mamba.summarize --artifact-root artifacts/smoke --expect-smoke`

Expected: completeness succeeds; every `dt_min >= 1e-3`, every `dt_max <= 1e-1`, and every finite flag is true.

- [ ] **Step 4: Commit any smoke-discovered code fixes one at a time**

For each failure, first add a regression test, prove it fails, apply one root-cause fix, rerun the focused test and full temporal suite, then commit with a message naming the root cause. Do not commit generated smoke artifacts.

---

### Task 12: Execute 36 Full Runs and Generate Evidence Report

**Files:**
- Generated: `artifacts/full/**`
- Generated/commit: `docs/experiment_report.md`

- [ ] **Step 1: Record the exact clean Git commit and environment**

Run: `git status --short` (expected empty) and `git rev-parse HEAD`.

Run remotely: `nvidia-smi` and `/home/lab/miniconda3/envs/evuav/bin/python -m torch.utils.collect_env`.

- [ ] **Step 2: Run the full matrix sequentially**

Run remotely:

```bash
python -m temporal_mamba.run_matrix \
  --datasets temporal_logic uci_har \
  --variants all \
  --seeds 42 123 777 \
  --artifact-root artifacts/full
```

Expected: 36 completed finite runs. A numerical failure triggers diagnosis and a tested fix before resuming; failed artifacts are retained for audit.

- [ ] **Step 3: Validate all completion evidence**

Run: `python -m temporal_mamba.summarize --artifact-root artifacts/full --expect-full`

Expected: strict 36/36 completeness, matching commit/config/dataset hashes, three-seed mean/std, paired deltas, temporal family metrics, HAR metrics, and time-order controls.

- [ ] **Step 4: Commit the generated evidence report**

```bash
git add docs/experiment_report.md
git commit -m "docs: report temporal mamba causal ablation"
```

Raw datasets, checkpoints, and run artifacts remain ignored and on Pro 6000; `experiment_report.md` includes their manifest hashes and remote paths.

---

### Task 13: GitHub Merge and Pro 6000 Deployment

**Files:**
- Git branch: `codex/temporal-mamba-ablation`
- Remote deployment: `/home/lab/projects/pcs3-temporal`

- [ ] **Step 1: Run final verification on the exact branch tip**

Run: `python -m pytest tests/test_temporal_*.py -q`

Run: `python -m temporal_mamba.summarize --artifact-root artifacts/full --expect-full`

Expected: all tests pass and 36/36 evidence validation passes.

- [ ] **Step 2: Push the verified feature branch**

Run: `git push -u origin codex/temporal-mamba-ablation`

Expected: GitHub branch points to the locally verified commit.

- [ ] **Step 3: Merge without rewriting master history**

Fetch origin, verify `origin/master` still descends from `ccc3fcab`, merge the feature branch into current master with a merge commit, rerun the focused test suite, and push master. If master moved, merge the updated origin/master into the feature branch and reverify before merging.

- [ ] **Step 4: Deploy the merged master to Pro 6000**

Clone or fast-forward `git@github.com:Newtonian-No/pcs3-phase1a.git` into `/home/lab/projects/pcs3-temporal`, verify the deployed `git rev-parse HEAD` equals GitHub master, and verify the report's artifact manifest hashes.

- [ ] **Step 5: Preserve the historical image project**

Run: `git -C /home/lab/projects/pcs3-temporal status --short` (expected empty). Confirm `/home/lab/projects/pcs3-phase1a` hashes for `ssm.py`, `pcs3_step2.py`, and `train_step2.py` still match the pre-work diagnostic record.
