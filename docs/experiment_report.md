# Temporal Mamba causal ablation report

Verified runs: 36/36

## Interpretation

- **Engineering objective achieved:** the experiment uses an explicit causal selective-SSM recurrence, a shared-parameter two-pass path, bounded error-conditioned `dt`, deterministic seeds, numerical guards, checkpoint recovery, and strict artifact validation.
- **Scientific objective not yet achieved:** on the held-out temporal-logic task, the four main variants remain near chance (0.5028-0.5161). The paired gains from a second pass, error injection, and auxiliary error supervision are small relative to their across-seed variation, so this matrix does not establish that the current model learns the intended temporal logic.
- **HAR is a functionality check, not sufficient proof of temporal reasoning:** the main variants reach 0.8888-0.8977 accuracy, but reversing time remains competitive (0.8930). The activity task therefore contains substantial cues that do not require the intended causal direction.
- **Order controls need careful interpretation:** shuffled/reversed temporal-logic labels are recomputed and become easier tasks; their high valid-label accuracy does not contradict the near-chance frozen-label and original-order results. They show sensitivity to the transformed task, not success on the original temporal-logic objective.
- **Hardware caveat:** random Python crashes were isolated to logical CPU 8 (and its sibling 9). All verified runs exclude CPUs 8-9 via process affinity; the workstation should be diagnosed independently before unrestricted workloads are trusted.

## temporal_logic

| Variant | Test accuracy mean | Sample std | Seeds |
|---|---:|---:|---:|
| vanilla | 0.502778 | 0.011276 | 3 |
| two_pass | 0.511250 | 0.006706 | 3 |
| error_inject | 0.513611 | 0.009189 | 3 |
| error_aux | 0.516111 | 0.011758 | 3 |
| time_shuffle | 0.792500 | 0.005204 | 3 |
| time_reverse | 0.865694 | 0.095512 | 3 |

Paired causal deltas:

- `two_pass-vanilla`: +0.008472 ± 0.010963
- `error_inject-two_pass`: +0.002361 ± 0.003592
- `error_aux-error_inject`: +0.002500 ± 0.003146

## uci_har

| Variant | Test accuracy mean | Sample std | Seeds |
|---|---:|---:|---:|
| vanilla | 0.888813 | 0.008146 | 3 |
| two_pass | 0.895261 | 0.005517 | 3 |
| error_inject | 0.892659 | 0.003041 | 3 |
| error_aux | 0.897749 | 0.006315 | 3 |
| time_shuffle | 0.867323 | 0.006474 | 3 |
| time_reverse | 0.892999 | 0.007296 | 3 |

Paired causal deltas:

- `two_pass-vanilla`: +0.006447 ± 0.006357
- `error_inject-two_pass`: -0.002602 ± 0.006618
- `error_aux-error_inject`: +0.005090 ± 0.009130

## Temporal formula families

| Variant | Seed | Family | Accuracy |
|---|---:|---|---:|
| error_aux | 123 | BEFORE | 0.555000 |
| error_aux | 123 | BOUNDED_RESPONSE | 0.575000 |
| error_aux | 123 | COUNT_WITHIN | 0.512500 |
| error_aux | 123 | EVENTUALLY | 0.560000 |
| error_aux | 123 | GAP | 0.480000 |
| error_aux | 123 | UNTIL | 0.495000 |
| error_aux | 42 | BEFORE | 0.492500 |
| error_aux | 42 | BOUNDED_RESPONSE | 0.497500 |
| error_aux | 42 | COUNT_WITHIN | 0.502500 |
| error_aux | 42 | EVENTUALLY | 0.535000 |
| error_aux | 42 | GAP | 0.510000 |
| error_aux | 42 | UNTIL | 0.510000 |
| error_aux | 777 | BEFORE | 0.525000 |
| error_aux | 777 | BOUNDED_RESPONSE | 0.477500 |
| error_aux | 777 | COUNT_WITHIN | 0.522500 |
| error_aux | 777 | EVENTUALLY | 0.510000 |
| error_aux | 777 | GAP | 0.482500 |
| error_aux | 777 | UNTIL | 0.547500 |
| error_inject | 123 | BEFORE | 0.522500 |
| error_inject | 123 | BOUNDED_RESPONSE | 0.515000 |
| error_inject | 123 | COUNT_WITHIN | 0.515000 |
| error_inject | 123 | EVENTUALLY | 0.562500 |
| error_inject | 123 | GAP | 0.505000 |
| error_inject | 123 | UNTIL | 0.522500 |
| error_inject | 42 | BEFORE | 0.510000 |
| error_inject | 42 | BOUNDED_RESPONSE | 0.477500 |
| error_inject | 42 | COUNT_WITHIN | 0.512500 |
| error_inject | 42 | EVENTUALLY | 0.535000 |
| error_inject | 42 | GAP | 0.495000 |
| error_inject | 42 | UNTIL | 0.505000 |
| error_inject | 777 | BEFORE | 0.525000 |
| error_inject | 777 | BOUNDED_RESPONSE | 0.485000 |
| error_inject | 777 | COUNT_WITHIN | 0.522500 |
| error_inject | 777 | EVENTUALLY | 0.532500 |
| error_inject | 777 | GAP | 0.465000 |
| error_inject | 777 | UNTIL | 0.537500 |
| time_reverse | 123 | BEFORE | 0.547500 |
| time_reverse | 123 | BOUNDED_RESPONSE | 1.000000 |
| time_reverse | 123 | COUNT_WITHIN | 0.475000 |
| time_reverse | 123 | EVENTUALLY | 0.510000 |
| time_reverse | 123 | GAP | 1.000000 |
| time_reverse | 123 | UNTIL | 1.000000 |
| time_reverse | 42 | BEFORE | 0.855000 |
| time_reverse | 42 | BOUNDED_RESPONSE | 1.000000 |
| time_reverse | 42 | COUNT_WITHIN | 0.782500 |
| time_reverse | 42 | EVENTUALLY | 0.880000 |
| time_reverse | 42 | GAP | 1.000000 |
| time_reverse | 42 | UNTIL | 1.000000 |
| time_reverse | 777 | BEFORE | 0.807500 |
| time_reverse | 777 | BOUNDED_RESPONSE | 1.000000 |
| time_reverse | 777 | COUNT_WITHIN | 0.840000 |
| time_reverse | 777 | EVENTUALLY | 0.885000 |
| time_reverse | 777 | GAP | 1.000000 |
| time_reverse | 777 | UNTIL | 1.000000 |
| time_shuffle | 123 | BEFORE | 0.490000 |
| time_shuffle | 123 | BOUNDED_RESPONSE | 0.955000 |
| time_shuffle | 123 | COUNT_WITHIN | 0.877500 |
| time_shuffle | 123 | EVENTUALLY | 0.462500 |
| time_shuffle | 123 | GAP | 0.945000 |
| time_shuffle | 123 | UNTIL | 0.990000 |
| time_shuffle | 42 | BEFORE | 0.520000 |
| time_shuffle | 42 | BOUNDED_RESPONSE | 0.955000 |
| time_shuffle | 42 | COUNT_WITHIN | 0.875000 |
| time_shuffle | 42 | EVENTUALLY | 0.480000 |
| time_shuffle | 42 | GAP | 0.945000 |
| time_shuffle | 42 | UNTIL | 0.990000 |
| time_shuffle | 777 | BEFORE | 0.507500 |
| time_shuffle | 777 | BOUNDED_RESPONSE | 0.950000 |
| time_shuffle | 777 | COUNT_WITHIN | 0.877500 |
| time_shuffle | 777 | EVENTUALLY | 0.512500 |
| time_shuffle | 777 | GAP | 0.942500 |
| time_shuffle | 777 | UNTIL | 0.990000 |
| two_pass | 123 | BEFORE | 0.535000 |
| two_pass | 123 | BOUNDED_RESPONSE | 0.515000 |
| two_pass | 123 | COUNT_WITHIN | 0.530000 |
| two_pass | 123 | EVENTUALLY | 0.552500 |
| two_pass | 123 | GAP | 0.467500 |
| two_pass | 123 | UNTIL | 0.505000 |
| two_pass | 42 | BEFORE | 0.505000 |
| two_pass | 42 | BOUNDED_RESPONSE | 0.487500 |
| two_pass | 42 | COUNT_WITHIN | 0.500000 |
| two_pass | 42 | EVENTUALLY | 0.535000 |
| two_pass | 42 | GAP | 0.507500 |
| two_pass | 42 | UNTIL | 0.490000 |
| two_pass | 777 | BEFORE | 0.530000 |
| two_pass | 777 | BOUNDED_RESPONSE | 0.475000 |
| two_pass | 777 | COUNT_WITHIN | 0.527500 |
| two_pass | 777 | EVENTUALLY | 0.520000 |
| two_pass | 777 | GAP | 0.477500 |
| two_pass | 777 | UNTIL | 0.542500 |
| vanilla | 123 | BEFORE | 0.497500 |
| vanilla | 123 | BOUNDED_RESPONSE | 0.442500 |
| vanilla | 123 | COUNT_WITHIN | 0.502500 |
| vanilla | 123 | EVENTUALLY | 0.510000 |
| vanilla | 123 | GAP | 0.540000 |
| vanilla | 123 | UNTIL | 0.502500 |
| vanilla | 42 | BEFORE | 0.515000 |
| vanilla | 42 | BOUNDED_RESPONSE | 0.487500 |
| vanilla | 42 | COUNT_WITHIN | 0.512500 |
| vanilla | 42 | EVENTUALLY | 0.480000 |
| vanilla | 42 | GAP | 0.490000 |
| vanilla | 42 | UNTIL | 0.477500 |
| vanilla | 777 | BEFORE | 0.562500 |
| vanilla | 777 | BOUNDED_RESPONSE | 0.460000 |
| vanilla | 777 | COUNT_WITHIN | 0.497500 |
| vanilla | 777 | EVENTUALLY | 0.547500 |
| vanilla | 777 | GAP | 0.515000 |
| vanilla | 777 | UNTIL | 0.510000 |

## Time-order controls

| Dataset | Variant | Seed | Valid-label accuracy | Frozen-label accuracy | Original-order accuracy |
|---|---|---:|---:|---:|---:|
| temporal_logic | time_shuffle | 42 | 0.794167 | 0.504167 | 0.502083 |
| temporal_logic | time_shuffle | 123 | 0.786667 | 0.495000 | 0.505417 |
| temporal_logic | time_shuffle | 777 | 0.796667 | 0.495833 | 0.500833 |
| temporal_logic | time_reverse | 42 | 0.919583 | 0.554167 | 0.624167 |
| temporal_logic | time_reverse | 123 | 0.755417 | 0.487500 | 0.492500 |
| temporal_logic | time_reverse | 777 | 0.922083 | 0.571667 | 0.610833 |
| uci_har | time_shuffle | 42 | 0.868001 | 0.868001 | 0.833051 |
| uci_har | time_shuffle | 123 | 0.873431 | 0.873431 | 0.784187 |
| uci_har | time_shuffle | 777 | 0.860536 | 0.860536 | 0.776043 |
| uci_har | time_reverse | 42 | 0.893112 | 0.893112 | 0.676620 |
| uci_har | time_reverse | 123 | 0.900238 | 0.900238 | 0.638616 |
| uci_har | time_reverse | 777 | 0.885646 | 0.885646 | 0.612827 |

## Provenance

- Git commit: `34ee48649da5780121f6dfbb80c0e455f4330e21`
- Dataset manifests: `{"temporal_logic": "941bc319b7f1f33384eaec2c3b62a0d7c625610875463f4e15aa8e57451d9c1e", "uci_har": "ff1d57a4af0ffae53c9d2ef96ec5eee1139a64eba568bba38baabb64813d88ee"}`
