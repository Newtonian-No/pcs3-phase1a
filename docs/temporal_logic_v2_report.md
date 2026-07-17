# Temporal Logic v2 experiment report

Verified runs: 12/12
Stage-2 gate: True

## Conclusion and scope

Scheme A removes the near-random failure: the query-bound models reach
99.79%--99.97% mean balanced accuracy on the in-distribution test set, versus
50.25% for the raw-concatenation diagnostic. Performance remains
99.81%--100.00% under channel permutation and 88.58%--89.85% on longer
sequences. Shuffling the temporal axis reduces accuracy to 49.58%--50.14%, and
reversing it reduces accuracy to 56.56%--58.83%, so the result depends on event
order rather than only on static channel statistics.

The four variants are effectively tied on the main test set; the error-aware
variants do not provide a material advantage over the simpler vanilla model in
this matrix. The current code therefore meets the controlled experiment goal:
a Mamba model can learn the six implemented temporal-logic families when the
query-to-channel binding is explicit and the rule distribution is matched.
It does not yet establish a general temporal-logic reasoner. Channel-OOD
invariance is partly enforced by the binder, and the experiment does not cover
unseen operators, compositional formulas, substantially longer horizons, or
real-world noise.

## Evaluation views

| View | Variant | Balanced accuracy mean | Sample std | Seeds |
|---|---|---:|---:|---:|
| val | vanilla | 1.000000 | 0.000000 | 3 |
| val | two_pass | 1.000000 | 0.000000 | 3 |
| val | error_inject | 0.998889 | 0.001925 | 3 |
| val | error_aux | 1.000000 | 0.000000 | 3 |
| test | vanilla | 0.999583 | 0.000417 | 3 |
| test | two_pass | 0.999444 | 0.000636 | 3 |
| test | error_inject | 0.997917 | 0.002917 | 3 |
| test | error_aux | 0.999722 | 0.000481 | 3 |
| long_test | vanilla | 0.885833 | 0.019094 | 3 |
| long_test | two_pass | 0.895417 | 0.035402 | 3 |
| long_test | error_inject | 0.898194 | 0.039822 | 3 |
| long_test | error_aux | 0.898472 | 0.020092 | 3 |
| channel_ood | vanilla | 1.000000 | 0.000000 | 3 |
| channel_ood | two_pass | 0.999722 | 0.000241 | 3 |
| channel_ood | error_inject | 0.998056 | 0.002646 | 3 |
| channel_ood | error_aux | 0.999861 | 0.000241 | 3 |
| reverse_frozen | vanilla | 0.573056 | 0.000241 | 3 |
| reverse_frozen | two_pass | 0.568333 | 0.006548 | 3 |
| reverse_frozen | error_inject | 0.565556 | 0.009075 | 3 |
| reverse_frozen | error_aux | 0.588333 | 0.026342 | 3 |
| shuffle_frozen | vanilla | 0.497639 | 0.003127 | 3 |
| shuffle_frozen | two_pass | 0.500139 | 0.005040 | 3 |
| shuffle_frozen | error_inject | 0.501389 | 0.006926 | 3 |
| shuffle_frozen | error_aux | 0.495833 | 0.000722 | 3 |

## Formula families

| Family | Variant | Test accuracy mean |
|---|---|---:|
| EVENTUALLY | vanilla | 1.000000 |
| EVENTUALLY | two_pass | 1.000000 |
| EVENTUALLY | error_inject | 1.000000 |
| EVENTUALLY | error_aux | 1.000000 |
| BEFORE | vanilla | 1.000000 |
| BEFORE | two_pass | 1.000000 |
| BEFORE | error_inject | 1.000000 |
| BEFORE | error_aux | 1.000000 |
| UNTIL | vanilla | 1.000000 |
| UNTIL | two_pass | 1.000000 |
| UNTIL | error_inject | 1.000000 |
| UNTIL | error_aux | 1.000000 |
| BOUNDED_RESPONSE | vanilla | 1.000000 |
| BOUNDED_RESPONSE | two_pass | 1.000000 |
| BOUNDED_RESPONSE | error_inject | 1.000000 |
| BOUNDED_RESPONSE | error_aux | 1.000000 |
| COUNT_WITHIN | vanilla | 0.997500 |
| COUNT_WITHIN | two_pass | 0.996667 |
| COUNT_WITHIN | error_inject | 0.987500 |
| COUNT_WITHIN | error_aux | 0.998333 |
| GAP | vanilla | 1.000000 |
| GAP | two_pass | 1.000000 |
| GAP | error_inject | 1.000000 |
| GAP | error_aux | 1.000000 |

## Attribution and order controls

- Binder minus raw: `0.4966666666666666`
- Reverse frozen minus ID: `-0.42534722222222215`
- Shuffle frozen minus ID: `-0.5004166666666666`
- Negative frozen-order deltas indicate sensitivity to temporal order under unchanged labels.

## Provenance

- Git commit: `fb064ca2cc9c04c179499c2e149cafef42eb23c7`
- Dataset manifest: `05e6ffad86fcb1702666101ab864f4d5c2b5e9fc523f358a54cbeeaabc9d6827`
