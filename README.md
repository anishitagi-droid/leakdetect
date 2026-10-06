# leakdetect — data leakage detector

Four composable layers built from the architecture doc. 78 tests; mutation-checked (see `REVIEW.md`).

## Run it
```
pip install scikit-learn pandas networkx scipy pytest
bash data/fetch.sh                                        # downloads the 4 example datasets (not stored in this repo)
pytest                                                    # 78 tests, ~15 s
python -m leakdetect data/titanic_seaborn.csv --target survived --positive 1
python -m leakdetect data/CreditCard.csv --target card --positive yes --drop rownames
python -m leakdetect data/pd_speech.csv --target class --positive PD --drop rownames \
       --script demo/pd_speech_leaky.py                   # dataset checks + static check of a modelling script
python demo/run_aer.py report                             # all four layers on AER (Layer 3 on a synthetic store)
python demo/run_examples.py report/examples               # Titanic, Parkinson's speech, stocks
```
The CLI exits 1 when a finding reaches `--fail-severity` (default 4), so it can gate CI. Layers 2 (runtime provenance)
and 3 (feature-store point-in-time) need live objects / a feature store and are used through the Python API (`demo/`).

| Layer | Detects | Input |
|---|---|---|
| 1 Static | fit/selection/encoding/oversampling before split, CV leakage, fit on test; follows local functions and methods | Python source, step list, sklearn Pipeline |
| 2 Runtime | contaminated fits via row lineage, exact + near duplicates across splits | instrumented estimators inside `with ProvenanceTracker()` |
| 3 PIT | wrong as-of joins, backfill leakage, online/offline skew | label / feature / training-set tables |
| 4 Stats | single-feature predictors, high mutual information, perfect-separation rules, too-similar splits | any CSV (NaNs/strings handled) |

## What it found (all numbers measured in this repo)
| dataset | finding | evidence |
|---|---|---|
| AER credit cards (1,319 x 11) | `expenditure`, `share` are post-outcome | RF AUC 0.994 -> 0.836 without them; `expenditure` is 0 for all 296 rejected rows |
| Titanic (891 x 15, raw CSV) | `alive` is a copy of the target (sev 5/4/3) | AUC 1.000 -> 0.859 without it; next-best feature (AUC 0.78) correctly not flagged |
| Parkinson's speech (252 x 750) | one severity-3 review item (`std_8th_delta`, a clean tail, AUC 0.77: probably a legitimately strong feature); the static check flags selection-before-split | 14 s runtime. The flagged bug costs +1.1pp (SE 0.4) on real labels but **+10.3pp on shuffled labels** (0.592 vs 0.489 honest) |
| Stocks, real prices | forward as-of join bug (sev 5, both features) | AUC 0.505 correct join vs 0.959 buggy (the bug is injected by me; the prices are real) |

On AER the same kind of pipeline bug costs only +0.1pp (n >> p): the layers find different things and a static finding is
not automatically a large effect.

## Limits
* Findings are evidence for review, not proof. Layer 4 cannot tell a leak from a legitimately dominant feature.
* Group/entity leakage with non-identical rows (several recordings of one patient) is NOT detected: Layer 2 finds
  exact and near-identical rows only. The Parkinson's file here has no subject id, so this was not tested.
* Layer 1 is straight-line analysis of recognised class names; Layer 2 needs instrumented estimators.
* Layer 3 has only been run on tables I built (synthetic AER store, stocks); not against Feast/Tecton.
* Layer 4's descriptive checks run on a median-imputed copy of the full data (a diagnostic, not an evaluation).

## How this was built
Implemented, tested and reviewed in conversation with Claude (Anthropic), working from an architecture document that
Anish supplied; Anish directed the scope and kept pushing for empirical validation over claimed results. Every number in
this README was produced by running the code in this repo on the datasets listed in `data/README.md`. `REVIEW.md` records
the defects that a code review and a second-dataset pass found and how each was fixed. The commit history groups one
build into logical steps; it is not a record of calendar time. Commits carry a `Co-authored-by: Claude` trailer.
No licence has been chosen yet.
