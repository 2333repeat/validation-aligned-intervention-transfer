# Evidence for the 600-policy extension

`protocol.json` fixes the scientific configuration. `model_manifest.json` identifies
all 600 terminal checkpoints and their SHA-256 digests. `data_manifest.json`
records thresholds, observation counts, dates and data hashes. These relative
artifact paths resolve after extracting the release asset into the repository root.

| File | Contents |
| --- | --- |
| `sensitivity.csv` | Hierarchical effect estimates by period and algorithm |
| `inference.csv` | Final estimates, intervals, bootstrap tail measures and BH adjustments |
| `alignment_methods.csv` | Eight correspondence rules on shared eligible records |
| `paired_method_comparisons.csv` | Paired comparisons of alternatives with Hungarian-abs |
| `native_diagnostics.csv` | Target-selected subset size and concept-specificity diagnostics |
| `economic_diagnostics.csv` | Descriptive original/target-selected financial differences |
| `policy_quality.csv` | Per-policy, period and transaction-cost financial diagnostics |
| `facts.json` | Coverage, aggregates and original experiment provenance |
| `PACKAGING.json` | Path metadata relocation and corresponding digest changes |
| `release.json` | Versioned artifact download URL, byte count and SHA-256 |

In `inference.csv`, `p_value` is the two-sided add-one bootstrap tail measure
computed from observed-data resamples. It is not an exact null-randomization
p-value. `q_value` applies Benjamini–Hochberg adjustment to those approximate
quantities. The families jointly cover both evaluation years, separately for
each resampling scheme: 4 primary sensitivity, 12 response correspondence,
2 shifted-alignment, and 4 all-concept contrasts. The algorithm family has 16
multiway contrasts; 28 paired mapping comparisons use a separate market-first
family. This documentation does not establish formal FDR control.

Response contrasts require both paired values to be finite. The conditional
action-redirection endpoint contains 22,610 / 22,577 primary records in
2025 / 2026; the other five response endpoints each contain 22,760 per year.
JSD fidelity differences reverse the distance subtraction so that positive
values favor the mapped response. Legacy column names containing `probability`
refer to normalized scores for DDQN; see the repository README.

The large artifact retains the per-record and observation-level arrays,
eligibility decisions, 50 random-subset controls and 50 shifted-alignment controls.
Its `MANIFEST.json` checks every payload. The release packaging does not change
the numerical CSV files, observations, weights, selected subsets or mapping arrays.
