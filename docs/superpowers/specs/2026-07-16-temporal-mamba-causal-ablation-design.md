# Temporal Mamba Causal Ablation Design

Status: approved in conversation on 2026-07-16

## 1. Purpose

Replace the current static-image patch experiment with a genuine temporal sequence experiment whose authoritative input contract is `B x T x D`. The new experiment must answer two separate questions:

1. Can a stable Mamba-style selective state-space encoder solve order-sensitive temporal logic and real multivariate sensor classification tasks?
2. After controlling for a second forward pass, what is the causal contribution of error-conditioned `dt`, the auxiliary prediction loss, and temporal order?

The work will create a new temporal package and experiment output tree. It will not overwrite the historical PC-S3 image code, logs, or checkpoints.

## 2. Scope and Non-Goals

### In scope

- True `B x T x D` signal tensors with time at dimension 1.
- A query-conditioned synthetic temporal-logic benchmark.
- Raw inertial-signal classification on UCI HAR.
- A directly recurrent selective SSM with auditable equations.
- Bounded, normalized error-conditioned `dt`.
- Six causal variants on both datasets and three seeds per variant: 36 full runs.
- Tests, resumable checkpoints, numerical diagnostics, aggregation, and a final evidence report.

### Out of scope

- Claiming that three seeds establish strong statistical significance.
- Hyperparameter tuning separately for each ablation variant.
- Reusing CIFAR patch order as a temporal axis.
- Replacing the direct recurrence with a custom parallel scan before correctness is established.
- Deleting or mutating `/home/lab/projects/pcs3-phase1a` historical artifacts.

## 3. Repository and Deployment Layout

Development occurs in a local Git repository created from the current remote source snapshot. New code is isolated under:

```text
temporal_mamba/
  ssm.py
  model.py
  losses.py
  metrics.py
  checkpoint.py
  datasets/
    temporal_logic.py
    uci_har.py
  train.py
  run_matrix.py
  summarize.py
tests/
  test_temporal_logic.py
  test_uci_har.py
  test_direct_scan.py
  test_temporal_model.py
  test_temporal_training.py
configs/
  temporal_logic.json
  uci_har.json
docs/
  experiment_report.md
```

After local and remote smoke verification, the repository is deployed to the sibling path `/home/lab/projects/pcs3-temporal`. The existing `/home/lab/projects/pcs3-phase1a` remains unchanged and acts as the historical reference.

Generated data, downloaded data, checkpoints, and run artifacts remain outside Git. Each run writes under `artifacts/<dataset>/<variant>/seed_<seed>/`.

## 4. Dataset A: Query-Conditioned Temporal Logic

### 4.1 Signal contract

Each sample contains:

- `signal`: `T x E`, where `E=8` binary event channels.
- `query`: a structured formula containing template ID, event arguments, and integer bounds.
- `label`: binary truth value produced by a deterministic reference evaluator.
- `sample_id`, `generation_seed`, and transformation metadata.

The model input is a single `T x D_in` tensor formed by concatenating:

1. the event signal;
2. normalized time `t/(T-1)`;
3. a query encoding broadcast to all time steps.

The label is never included in the input. Batching produces `B x T x D_in`. The default training length is 128. Correctness tests cover lengths 64, 128, and 256, and a separate length-generalization test set uses length 256.

### 4.2 Formula families

The benchmark uses six formula families:

1. `EVENTUALLY(A)`: event A occurs at least once.
2. `BEFORE(A, B)`: an A occurs before the first B, and B occurs.
3. `UNTIL(A, B)`: B occurs and A is active at every step before the first B.
4. `BOUNDED_RESPONSE(A, B, k)`: every A is followed by B within `k` steps.
5. `COUNT_WITHIN(A, start, end, m)`: A occurs at least `m` times in the bounded interval.
6. `GAP(A, B, low, high)`: some A is followed by B after a gap in the inclusive range.

The evaluator is the single source of truth. The generator constructively creates positives and negatives, then verifies every label with the evaluator. Class balance is 50/50 within each formula family, allowing a deviation of at most one sample.

### 4.3 Splits and leakage prevention

- A dataset manifest is generated once with fixed `data_seed=20260716` and reused by every model variant and training seed. Train, validation, in-distribution test, and length-generalization test use disjoint child seeds derived from that fixed data seed.
- Sample IDs and generated event sequences are checked for exact duplicates across splits.
- Formula template and argument distributions are stratified across splits.
- Query parameters are recorded in artifacts so per-family metrics can be reproduced.

### 4.4 Temporal transformations

- `time_reverse`: reverse only the dynamic signal, rebuild the normalized time channel, retain the query, and recompute the truth label with the evaluator.
- `time_shuffle`: apply a deterministic per-sample time permutation to the dynamic signal, rebuild time, retain the query, and recompute the truth label.

For diagnostic evaluation, transformed inputs are also scored against the original frozen labels. The report therefore distinguishes valid transformed-task accuracy from sensitivity to order changes.

## 5. Dataset B: UCI HAR Raw Inertial Signals

The real-world benchmark is the UCI Human Activity Recognition Using Smartphones dataset, DOI `10.24432/C54S4K`, licensed CC BY 4.0.

- Use the nine raw inertial channels from `Inertial Signals`, not the precomputed 561-feature vectors.
- Each sample is `T=128`, `D=9`.
- Use the official subject-disjoint train/test split and six activity labels.
- Create validation data from training subjects only; no test subject enters model selection. Validation subject IDs are selected once with fixed `data_seed=20260716`, persisted in the dataset manifest, and reused by all runs.
- Compute channel mean and standard deviation from the training split only, then reuse them for validation and test.
- Persist the source URL, archive SHA-256, extraction manifest, and attribution in the run metadata.

For HAR temporal controls, reversal and shuffling preserve the activity label. This intentionally tests whether models rely on temporal order or mostly on order-insensitive signal statistics.

## 6. Stable Direct Selective SSM

### 6.1 Base recurrence

For input `u` with shape `B x T x D_inner`, each layer computes input-dependent `B_t`, `C_t`, and base `dt_t`, then uses the exact recurrence:

```text
A = -exp(A_log)
A_bar_t = exp(dt_t * A)
B_bar_t = dt_t * B_t
h_t = A_bar_t * h_(t-1) + B_bar_t * u_t
y_t = sum(C_t * h_t, state_dim) + D_skip * u_t
```

The recurrence is a Python loop over `T` containing tensor operations over batch, channel, and state dimensions. With `T <= 256`, correctness and auditability take priority over a parallel kernel.

### 6.2 Initialization

- `A_log[d, n] = log(n + 1)` for state indices `n=0..N-1`.
- `dt_rank = ceil(D_inner / 16)`.
- `dt_proj.weight` follows the official Mamba scale of `dt_rank^-0.5`.
- Initial `dt` values are log-uniform in `[dt_min, dt_max]`; the projection bias is set with inverse softplus.
- Defaults: `dt_min=1e-3`, `dt_max=1e-1`.
- `D_skip` initializes to one.
- The recurrence and exponentials run in float32 even if future outer layers use automatic mixed precision.

### 6.3 Error-conditioned dt

The temporal prediction error has raw shape `B x T x (2*D_signal)` and contains aligned position and velocity errors. It passes through RMS normalization and a learned projection. Modulation is:

```text
alpha = alpha_max * tanh(alpha_raw)
modulation = alpha * tanh(error_projection(normalized_error))
dt = clamp(dt_base * exp(modulation), dt_min, dt_max)
```

`alpha_raw` is initialized to zero, so the model begins as the stable base SSM and learns modulation gradually. `alpha_max=log(4)` bounds the multiplicative modulation before final clamping.

Every forward pass records aggregated `dt_min`, `dt_max`, error RMS/max, SSM output RMS/max, and finite-status diagnostics. Per-batch detailed traces are enabled only for failed runs or explicit diagnostic mode.

## 7. True Temporal Prediction and Two-Pass Data Flow

### 7.1 First pass

The embedded temporal input is processed without error modulation. Final hidden state `H1[:, t]` predicts the next dynamic signal and its velocity:

```text
pred_x[:, t] -> signal[:, t+1]
pred_v[:, t] -> signal[:, t+1] - signal[:, t]
```

Aligned errors are zero at `t=0`; for `t>=1`:

```text
position_error[:, t] = signal[:, t] - pred_x[:, t-1]
velocity_error[:, t] = (signal[:, t] - signal[:, t-1]) - pred_v[:, t-1]
```

Targets and `H1` are detached at the predictor boundary. This prevents the auxiliary predictor and the second-pass error path from directly backpropagating through the first-pass activation graph. The shared encoder parameters still learn from the classification loss through the second pass.

### 7.2 Second pass

The same encoder parameters process the original embedded input again from a zero initial state.

- `two_pass` supplies no error modulation.
- `error_inject` supplies normalized temporal errors but sets auxiliary weight to zero.
- `error_aux` supplies the same errors and adds robust auxiliary loss.

The classification head always consumes the final valid time step after the encoder's output normalization. Temporal-logic uses one binary logit; HAR uses six logits.

## 8. Auxiliary Loss

Auxiliary loss uses elementwise Smooth L1 for position and velocity, excludes the undefined first time step, then averages the already-computed penalties:

```text
L_aux = mean(smooth_l1(position_error[:, 1:]))
      + velocity_weight * mean(smooth_l1(velocity_error[:, 1:]))
L_total = L_task + lambda_aux * L_aux
```

Errors are never averaged across time or prediction points before applying the penalty. `lambda_aux` warms linearly from zero over the first 10% of epochs to its configured value. `error_inject` always uses `lambda_aux=0`.

## 9. Causal Ablation Matrix

The six variants are:

| Variant | Passes | Error-conditioned dt | Auxiliary loss | Temporal transform |
|---|---:|---:|---:|---|
| `vanilla` | 1 | no | no | none |
| `two_pass` | 2 | no | no | none |
| `error_inject` | 2 | yes | no | none |
| `error_aux` | 2 | yes | yes | none |
| `time_shuffle` | 2 | yes | yes | deterministic shuffle |
| `time_reverse` | 2 | yes | yes | reverse |

Each variant runs with seeds `42`, `123`, and `777` on both datasets: `6 x 3 x 2 = 36` full runs.

Within a dataset, architecture size, optimizer, training schedule, batch size, and early-stopping rule are frozen across variants. A single vanilla-only pilot may select dataset-level hyperparameters before the matrix starts; pilot results are labeled and excluded from the matrix.

## 10. Training and Reproducibility

- Optimizer: AdamW with dataset-level fixed learning rate and weight decay.
- Schedule: linear warmup for 5% of total steps, then cosine decay.
- Gradient clipping uses `error_if_nonfinite=True`.
- Loss, parameters, gradients, activations, and `dt` are checked for finite values.
- A non-finite batch aborts the run, writes a diagnostic artifact, and leaves the last healthy atomic checkpoint. It is not silently skipped.
- Checkpoints contain model, optimizer, scheduler, scaler if present, RNG states, epoch, best metric, configuration, dataset manifest, and Git commit.
- Resumption restores all recorded state and appends to the same run history.
- Training seeds `42`, `123`, and `777` configure Python, NumPy, PyTorch, CUDA, model initialization, dropout, and data-loader generators. Dataset splits and per-sample temporal transformations come from the fixed dataset manifest, so paired variant comparisons see identical source samples.
- Deterministic algorithms are used where supported; any unavoidable nondeterministic operator is recorded.

## 11. Metrics and Comparisons

### Temporal logic

- Accuracy, balanced accuracy, and F1.
- Accuracy and F1 per formula family.
- In-distribution and length-256 generalization metrics.
- Recomputed-label metrics on transformed sequences.
- Frozen-label degradation and truth-flip accuracy under reversal.

### UCI HAR

- Accuracy, macro-F1, per-class recall, and confusion matrix.
- Accuracy degradation under shuffle and reversal relative to `error_aux`.

### Across seeds

- Mean and sample standard deviation for every primary metric.
- Paired seed deltas for `two_pass - vanilla`, `error_inject - two_pass`, and `error_aux - error_inject`.
- No unsupported p-value or superiority claim from only three seeds.

## 12. Tests and Verification Gates

### Unit tests

1. Direct recurrence matches an independent reference recurrence in float32 and float64.
2. Gradients of the direct recurrence match the reference on a small problem.
3. `dt` remains inside configured bounds under extreme finite errors.
4. Zero-initialized error modulation exactly matches base `dt`.
5. Temporal formula evaluators pass hand-constructed positive and negative cases.
6. Generated datasets are balanced, deterministic, duplicate-free across splits, and label-verified.
7. Reverse and shuffle transforms produce expected signals and recomputed labels.
8. UCI HAR loader returns `B x 128 x 9`, respects subject splits, and uses train-only normalization.
9. All six model variants honor their pass, error, loss, and transform contracts.
10. Auxiliary loss cannot cancel errors by averaging before Smooth L1.
11. A checkpoint/resume round trip reproduces the next optimization step.
12. Numerical guard tests intentionally trigger and capture non-finite failure artifacts.

### Integration gates

1. Overfit a tiny batch for both datasets.
2. Run one-epoch smoke tests for all six variants on both datasets.
3. Confirm every smoke artifact includes configuration, metrics, diagnostics, and checkpoint metadata.
4. Run all 36 full experiments without non-finite values.
5. Regenerate the aggregate summary and report from raw run artifacts.

## 13. Completion Criteria

The project is complete only when all of the following are true:

1. The remote deployed code accepts and validates genuine `B x T x D` inputs.
2. Direct scan correctness, gradient, initialization, and bounded-error tests pass.
3. Temporal-logic and UCI HAR loaders satisfy their documented split and shape contracts.
4. All six ablation variants are implemented as specified, not inferred from configuration names alone.
5. All 36 runs have final artifacts and finite diagnostics.
6. The summary contains three-seed mean/std and paired causal deltas.
7. Time shuffle and reversal outcomes are reported for both datasets.
8. The final report distinguishes implementation stability, task performance, and causal interpretation, including negative results.
9. Remote file hashes and the recorded Git commit identify the exact code used for every run.

No performance improvement is required to call the experiment executed successfully. If error conditioning does not improve either benchmark, that negative result must be reported rather than hidden or retuned per variant.
