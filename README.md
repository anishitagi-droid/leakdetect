# leakdetect

[![CI](https://github.com/anishitagi-droid/leakdetect/actions/workflows/ci.yml/badge.svg)](https://github.com/anishitagi-droid/leakdetect/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A data-leakage detector for tabular machine learning. Leakage makes a model look better than it is: a feature that
secretly encodes the answer, a scaler fitted before the train/test split, an entity that appears on both sides of a
split. `leakdetect` looks for it from four angles, and reports **evidence with severities and measured impact** instead
of a verdict.

```
$ leakdetect data/titanic_seaborn.csv --target survived --positive 1
# Leakage report
**1 finding(s)** after de-duplication (3 raw), max severity **5/5**. Layers run: [4].

## [5/5] alive_yes — single-feature-predictive-power + high-mutual-information + perfect-separation-rule
**Evidence:** 'alive_yes' alone reaches CV AUC 1.000 (monotone 1.000, depth-2 tree 1.000); full 5-fold random forest AUC is 1.000. …
**Fix:** Check whether 'alive_yes' is observable before the label exists. Drop it, or refit and report metrics without it.
```
(`alive` in this copy of the Titanic data is a literal copy of the target. Evidence line shortened here.)

## Install and run
```
pip install git+https://github.com/anishitagi-droid/leakdetect       # or: git clone ... && pip install -e ".[dev]"
leakdetect data.csv --target y --positive 1 --drop row_id            # dataset checks (Layer 4)
leakdetect data.csv --target y --positive 1 --group patient_id       # + entity leakage
leakdetect data.csv --target y --positive 1 --script train.py        # + static check of the modelling script (Layer 1)
```
Exit status is 1 when a finding reaches `--fail-severity` (default 4), so it can gate CI. Layers 2 and 3 need live
Python objects or a feature store and are used through the Python API (see `demo/`).

To reproduce every number below: `bash data/fetch.sh` (downloads 7 public datasets and verifies SHA-256 checksums; the
data is not stored here), then `python demo/run_aer.py report` and `python demo/run_examples.py report/examples`.

## The four layers
| Layer | Detects | Input |
|---|---|---|
| 1 Static | fit / selection / encoding / oversampling before the split, CV leakage, fit on test data; follows local functions and methods | Python source, step list, sklearn Pipeline |
| 2 Runtime | contaminated fits via row-level lineage, exact and near-duplicate rows across splits | instrumented estimators inside `with ProvenanceTracker()` |
| 3 Point-in-time | wrong as-of joins, backfill leakage, online/offline skew in feature stores | label, feature and training-set tables |
| 4 Statistical | single-feature predictors, high mutual information, perfect-separation rules, too-similar splits, entity leakage, and how far AUC falls without the flagged features | any CSV (NaNs and strings handled) |

## What it found (all numbers measured in this repo)
| dataset | finding | evidence |
|---|---|---|
| AER credit cards (1,319 x 11) | `expenditure`, `share` are post-outcome | RF AUC 0.994 -> 0.836 without them (10-seed mean; the tool's own single-seed estimate: 0.995 -> 0.829); `expenditure` is 0 for all 296 rejected rows |
| Titanic (891 x 15, raw CSV) | `alive` is a copy of the target (sev 5/4/3) | AUC 1.000 -> 0.859 without it; the next-best feature (AUC 0.78) is correctly not flagged |
| Parkinson's speech (252 x 750) | one severity-3 review item (`std_8th_delta`, a clean tail, AUC 0.77: probably a legitimately strong feature); the static check flags selection-before-split | 14 s runtime. The flagged bug costs +1.1pp (SE 0.4) on real labels but **+10.3pp on shuffled labels** (0.592 vs 0.489 honest) |
| Orthodont, ChickWeight, VerbAgg (repeated measures) | 100% of test rows share an entity with train under a random split, yet random-vs-grouped AUC gaps are -0.024, +0.005, -0.002: nothing flagged, correctly | a synthetic memorisable-subjects control **is** flagged (random 0.891 vs grouped 0.419); 0 of 10 shuffled-id controls are |
| Stocks, real prices | forward as-of join bug (sev 5, both features) | AUC 0.505 with the correct join vs 0.959 with the bug (the bug is injected; the prices are real) |

On AER the same kind of pipeline bug costs only +0.1pp (n >> p), so a static finding is not automatically a large effect.

## Design decisions
* **Absolute, multiplicity-aware thresholds.** The first mutual-information rule ("z >= 2 above the other features")
  flagged 42 of 750 *pure-noise* features. It is now "carries >= 50% of the target's entropy": 0 flags on the same noise.
  The perfect-separation rule Bonferroni-corrects over features x thresholds, and severity depends on how much of a class
  the rule covers.
* **"Skipped" is never "clean".** Every layer returns an explicit skipped status with a reason; the report lists layers
  that did not run.
* **Evidence over assertion.** Findings come with the AUC change when the feature is removed, and each detector was run
  against controls where the right answer is known (shuffled labels, shuffled entity ids, pure noise).
* **Safe instrumentation.** Instrumented estimators are exact no-ops outside a tracker context; an early version kept state
  in a hidden global (found in review, fixed).
* **Fast paths are checked against references.** The vectorised KS test equals `scipy.stats.ks_2samp`, including heavy
  ties; Layer 4 on 252 x 750 went from 139 s to 14 s.

## Testing
100 tests, run in CI on Python 3.11-3.13 with the latest dependencies and on the oldest supported ones
(pandas 2.2.3, scikit-learn 1.5.2, numpy 1.26.4, scipy 1.13.1, networkx 3.3). Beyond unit tests, bugs were injected one at
a time to check the suite catches them (mutation checks: 14/14, 12/12 and 9/9 killed on green baselines), and the demo output
is diffed against a saved baseline after refactors. `REVIEW.md` is the engineering log: every defect found, how, and the fix,
including two of my own mistakes. The suite takes about 85 s on one CPU because forest fits dominate.

## Limits
* Findings are evidence for review, not proof. Layer 4 cannot tell a leak from a legitimately dominant feature.
* Entity leakage is detected only through its effect on AUC (`--group`), and the only case where it fires is a synthetic
  positive control: on three real repeated-measures datasets it reports overlap but no inflation. It flags a gap >= 0.03
  and > 3 SE; at 60 subjects x 6 rows, gaps of about 0.05 were not flagged. Binary targets only. Layer 2 still finds only
  exact or near-identical rows. The Parkinson's file has no subject id.
* Layer 1 is straight-line analysis of recognised class names; Layer 2 needs instrumented estimators.
* Layer 3 has only been run on tables built for the demos (a synthetic AER store, stocks); not against Feast or Tecton.
* Layer 4's descriptive checks run on a median-imputed copy of the full data (a diagnostic, not an evaluation).

## How this was built
Designed and implemented by Claude (Anthropic): the four-layer architecture, every module and method in `leakdetect/`, the
tests and the demos. Anish set the goal and the standard: he supplied the starting brief, insisted that every claim be backed
by an experiment on real data, asked for the code review, the extra datasets and the fuzzing that found most of the defects
listed in `REVIEW.md`, and decided what to publish. Every number in this README was produced by running the code in this
repo on the datasets listed in `data/README.md`. The commit history groups one build into logical steps; it is not a record
of calendar time, and commits carry a `Co-authored-by: Claude` trailer.

## Layout
`leakdetect/` the package (one module per layer, plus `aggregator.py` and the CLI) · `tests/` · `demo/` runnable examples
and the leaky/clean scripts used for Layer 1 · `data/` checksummed fetch script and dataset provenance · `REVIEW.md`
engineering log · MIT licensed.
