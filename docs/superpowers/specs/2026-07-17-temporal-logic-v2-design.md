# Temporal Logic v2 Design

Date: 2026-07-17

## 1. Context and diagnosis

The first temporal-logic matrix completed 36/36 runs with finite numerics, but the four main variants reached only 0.5028-0.5161 held-out accuracy. This was not an optimization failure: vanilla training accuracy reached 0.7555 while validation stayed at 0.5004, and error-aux training accuracy reached 0.9409 while validation stayed at 0.5221. Every formula family remained close to chance.

The primary structural problem is query-to-channel binding. A sample supplies an event tensor plus one-hot identities for events A and B, but the current model receives them as concatenated features and must discover the multiplicative operation that selects the queried event channels. The model can memorize random training instances without learning that binding rule. A secondary benchmark problem is that recomputing labels after reversal or shuffling produces severely imbalanced transformed tasks, so high transformed-label accuracy is not a valid temporal-reasoning result.

## 2. Goals

1. Preserve the v1 dataset, artifacts, report, and code path as an immutable baseline.
2. Add a v2 benchmark whose positive and negative examples are balanced and matched on nuisance statistics.
3. Make query-to-channel binding explicit while leaving the temporal rule itself for the causal Mamba to learn.
4. Preserve the shared-weight one-pass/two-pass/error-injection/error-auxiliary ablation.
5. Require held-out, per-family, and channel-permutation generalization before starting a full matrix.
6. Replace transformed-label training controls with balanced, frozen-label evaluation controls.

## 3. Non-goals

1. Do not replace the Mamba/SSM backbone with attention or a rule engine.
2. Do not encode formula truth values or handcrafted formula outputs as input features.
3. Do not tune against the test or OOD splits.
4. Do not rerun UCI HAR as a full matrix; only a regression smoke check is required because v2 changes are isolated to the temporal-logic path.
5. Do not overwrite `artifacts/full` or the v1 dataset manifest.

## 4. Dataset design

### 4.1 New dataset and splits

Add an independent `temporal_logic_v2` dataset with deterministic train, validation, test, long-test, and channel-OOD splits. Each split must have equal counts for every `(formula_family, label)` pair, no cross-split fingerprints, immutable file hashes, and a canonical manifest hash.

The v2 dataset returns raw event signals and a structured query separately. It must not broadcast the query into every raw feature vector. The structured query contains the family one-hot vector, event-A one-hot vector, event-B one-hot vector, and normalized integer parameters.

The channel-OOD split uses event identities and channel permutations not used by the in-distribution training generator. Because binding is deterministic, performance should be invariant to channel names; this split verifies that the data-to-model binding path is correct and that no raw channel ID leaks into the decision.

### 4.2 Matched positive and negative construction

All formula families keep equal global event counts and matched position marginals wherever possible:

- `EVENTUALLY`: both labels contain the same number of A events; positives place the decisive event inside the query window and negatives place it outside.
- `BEFORE`: both labels contain one decisive A and B with matched sampled positions; only their order changes.
- `UNTIL`: positives contain a continuous A prefix before B. Negatives remove one prefix A and add a compensating A after B so global counts remain matched.
- `BOUNDED_RESPONSE`: both labels contain the same A triggers and B responses. A negative moves exactly one response just outside its permitted horizon.
- `COUNT_WITHIN`: both labels contain the same global number of A events. Positives meet the in-window threshold; negatives move one A from inside to outside the window.
- `GAP`: both labels contain the same number of A and B events. Positives use an in-range gap; negatives use a nearest-boundary out-of-range gap.

Decisive positions, windows, horizons, and gaps are sampled across their valid ranges rather than placed at fixed fractions. Irrelevant channels use the same distractor distribution for both labels. Every sample is verified by the existing temporal-query evaluator before it is persisted.

### 4.3 Time-order controls

Reversal and deterministic shuffling become evaluation views only. They retain the original balanced labels and are evaluated by a model trained on original-order data. The report records original, reversed-frozen-label, and shuffled-frozen-label accuracy overall and per formula family. No full model is trained on a recomputed, imbalanced transformed-label dataset.

## 5. Model design

### 5.1 Explicit query binder

Add a deterministic, differentiable query binder at the temporal-logic model boundary:

- `a_t = dot(raw_signal_t, event_a_one_hot)`
- `b_t = dot(raw_signal_t, event_b_one_hot)`, or zero when B is absent
- normalized time `t`
- signed normalized offsets from query parameters `p0` and `p1`

The binder emits time-varying streams only. It does not evaluate a temporal formula. Formula family and normalized parameters remain a separate global query condition.

This canonicalization removes the unlearned channel-selection operation while preserving the temporal work: the model must still detect windows, order, persistence, bounded response, counts, and gaps.

### 5.2 Query-conditioned causal Mamba

A small query conditioner maps the family one-hot vector and normalized parameters to per-layer scale and shift values. Each Mamba block applies bounded FiLM conditioning after normalization and before its causal convolution/SSM path. Scale values are bounded around one and shifts are bounded around zero to preserve numerical guards.

The existing direct selective recurrence, float32 state update, bounded `dt`, zero-initialized error modulation, and shared encoder modules remain unchanged unless a failing regression test proves a required interface adjustment.

### 5.3 Readout

The classifier receives a concatenation of:

- the final normalized hidden state;
- a running-prefix maximum summarized at the final step;
- a running-prefix mean summarized at the final step.

These summaries are computed only from states at or before each time index. The final sequence classifier therefore remains causal while gaining robust access to sparse and persistent events.

### 5.4 Error pathway

The next-step predictor targets only the query-bound A/B streams and their first differences, not all raw distractor channels. The two-pass encoder continues to share all Mamba parameters. Variants retain their meanings:

- `vanilla`: one causal pass, no error input;
- `two_pass`: two shared-parameter passes, no error input;
- `error_inject`: two passes with bounded error-conditioned `dt`;
- `error_aux`: error injection plus the warmed auxiliary prediction loss.

## 6. Ablations and attribution

Before the v2 gate, run a single-seed `raw_concat` diagnostic using the v1-style concatenated input on v2 data. This is not part of the full matrix; it tests whether explicit binding is the material change. The primary full matrix remains four variants by three seeds.

All comparisons use identical v2 manifests and paired seeds. The report includes:

- explicit binder versus raw concatenation;
- `two_pass - vanilla`;
- `error_inject - two_pass`;
- `error_aux - error_inject`;
- original versus frozen-label reverse/shuffle evaluation.

## 7. Validation and experiment stages

### Stage 0: deterministic correctness

1. Unit tests cover every matched positive/negative constructor.
2. Manifest tests prove balance, uniqueness, reproducibility, and channel-OOD separation.
3. Binder tests prove exact A/B extraction and channel-permutation invariance.
4. An oracle evaluation of persisted labels must be 100%.
5. Existing v1 and UCI HAR tests must continue to pass.

### Stage 1: optimization smoke gates

1. A tiny v2 batch must overfit to at least 98% accuracy.
2. All logits, gradients, recurrent states, losses, and metrics must remain finite.
3. Observed `dt` must remain in `[1e-3, 1e-1]`.
4. One UCI HAR smoke run confirms the generic model path did not regress.

### Stage 2: two-seed generalization gate

Run bound `vanilla` with seeds 42 and 123. Both seeds must satisfy all of:

- validation balanced accuracy at least 0.80;
- every formula-family validation accuracy at least 0.70;
- channel-OOD balanced accuracy at least 0.70;
- finite diagnostics and valid `dt` bounds.

If either seed fails, preserve its artifacts and stop. Form one new root-cause hypothesis and test one change; do not launch the full matrix.

### Stage 3: full v2 matrix

Only after Stage 2 passes, run 12 jobs:

- variants: `vanilla`, `two_pass`, `error_inject`, `error_aux`;
- seeds: 42, 123, 777.

Each completed run evaluates validation, test, long-test, channel-OOD, reverse-frozen-label, and shuffle-frozen-label views. A strict summarizer rejects missing jobs, mixed code commits, mixed manifests, non-finite values, or invalid `dt` ranges.

## 8. Artifacts and provenance

Use separate roots:

- dataset: `/home/lab/datasets/pcs3-temporal/temporal_logic_v2`;
- smoke artifacts: `artifacts/v2-smoke`;
- gate artifacts: `artifacts/v2-gate`;
- full artifacts: `artifacts/v2-full`.

Every final artifact records the exact Git commit, config hash, dataset manifest hash, CPU affinity, GPU identity, PyTorch/CUDA versions, seed, variant, and split metrics. All Pro 6000 Python workloads exclude logical CPUs 8 and 9 because the first experiment isolated random interpreter crashes to that physical core pair.

## 9. Failure handling

- Dataset validation failures abort before training.
- Numerical failures write a structured failure artifact and retain the last healthy checkpoint.
- Gate failures stop before the full matrix and retain all diagnostic evidence.
- Matrix resumption reuses a result only when run ID, config hash, dataset manifest hash, and Git commit all match.
- v1 datasets, artifacts, and report are never mutated.

## 10. Acceptance criteria

The implementation is ready to launch the full v2 matrix only when Stage 0, Stage 1, and both Stage 2 seeds pass. The task is scientifically successful only when the strict 12/12 summary passes and the resulting report distinguishes in-distribution temporal reasoning, per-family behavior, channel-OOD generalization, and frozen-label order sensitivity.
