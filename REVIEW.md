# Code review — leakdetect (senior-engineer pass)

Line numbers refer to the code **before** this review (snapshot kept outside the repo). Tooling used: ruff
(F,E,B,SIM,C4,PIE,RET,ARG,UP,RUF,PERF), vulture, radon, a golden-output diff of the demo, and a hand-rolled
mutation check. Every change below was applied; "Not changed" lists what was deliberately left.

## A. Defects (behaviour was wrong or misleading)
| # | Where | Problem | Resolution |
|---|---|---|---|
| A1 | layer2 L165-170 `current_tracker()` | Created and *installed* a tracker when none was active. Instrumented estimators then recorded every fit forever in an invisible global (3 fits retained 3,000 row hashes — verified). | Wrappers are a pass-through outside `with ProvenanceTracker()`. `tracked_train_test_split` raises `RuntimeError` instead of hiding state. Tests added. |
| A2 | layer1 `visit_FunctionDef` (+ class bodies) | A leak inside a class method returned `skipped: no train_test_split recognised` — wrong reason, silent false negative. | Methods are analysed at their call site like functions. 3 tests added. |
| A3 | layer1 L391 note | Said functions are "not modelled" — false since inlining was added. | Note rewritten to the actual limits. |
| A4 | tests `test_stale_join_...` | Never asserted a stale join was found; passed with the detection deleted. Computed an unused `ts`. | Asserts `stale-join` present and `pit-violation` absent. |
| A5 | demo L60 | `y[Xd.index[len(X):] % len(X)]` labelled injected duplicates with the wrong rows' labels. (No effect on printed results: duplicate detection ignores y.) | Uses the sampled rows' own labels. |
| A6 | categories / layer1 L122 | `RFECV` was in both `SELECTION` and `SEARCH_CV`; `_call` papered over it with `and category_of(cls) is None`. | Removed from `SEARCH_CV`; the patch condition is gone. |
| A7 | layer1 `analyze_pipeline(assume_fit_on_train_only=False)` | The only non-default value could only ever return "skipped". | Parameter removed. |
| A8 | aggregator `to_json(default=str)` | Silently stringified anything unserialisable, hiding bugs. | Removed; findings are plain str/int. |

## B. Performance
| # | Where | Problem | Result |
|---|---|---|---|
| B1 | layer3 L84-97 | Python loop over mismatched rows doing a pandas boolean-index each, plus `pd.Series([joined]*len(g))` per row (L90). | One merge + groupby. **1.06s → 0.03s (1k entities), 3.76s → 0.05s (4k)**, identical findings (evidence-hash equal). |
| B2 | layer1 L257-261, L274, L338, L351 | `nx.ancestors` recomputed per (fit, split) pair and per (CV, fit) pair; `desc_of_cv` rebuilt inside a loop. | Computed once per node. |
| B3 | layer2 L47-48, L137, L99-100 | `df.copy()` + column rename before hashing (hash is label-independent — verified); roots stored as `uint64` array then rebuilt into a set on every check; per-element `int()` calls. | No copy; `frozenset` stored once; `.tolist()`. |
| B4 | layer3 L62 | Copied the entire (wide) training set to convert one column. | Key-only frame. |

## C. Dead or wasteful code removed
aggregator `_FAMILY_FOR_STEP` (never read), unused `LayerResult` import, `LayerResult.to_dict`;
layer1 `res.graph`, `step=i` node attr, `FIT_KINDS_ALL`, always-true `(is_attr or isinstance(..., ast.Name))` (L141),
`set().union(...) if (...) else set()` (L100), four one-line rule wrappers (now a `partial` table), and the
**`transform` branch in `_call`, found dead by mutation testing** (no rule reads it; data flows via the fallback);
layer2 `FitEvent.eid` / `.hashes` (stored, never read), `field` import, `max_brute` parameter and its cryptic gate
(`ntr*nte <= max_brute**2//4 and A.shape[0] <= max_brute`), duplicated `X = args[0] if args else ...` blocks;
layer3 unused `features=` parameter, a per-feature stats dict whose fields nobody read, ternary-for-side-effects
(L181/183); layer4 unused `rng` (L161); demo `sy_leak = ... if False else None` (L103), `E741 l`, `;` one-liners,
`open()` without a context manager. Undeclared attributes (`res.tables`, `res.graph`) replaced by a declared
`LayerResult.extras`.

## D. Structure / readability
`_call` cyclomatic complexity 36 → 24 (`_node` helper replaces five copies of create/link/return; fit branch
extracted). `find_cross_split_duplicates` and `check_point_in_time` split into focused helpers. Re-entrancy guard
moved into the tracker (`tracker.call`) instead of module-level code poking `t._depth` / `t._on_fit`. Lambda +
`functools.wraps` instance hack replaced by `types.MethodType`. Broad `except Exception` narrowed. Inline imports
hoisted (the lazy sklearn import in `analyze_pipeline` is kept on purpose: source analysis needs no sklearn).
Magic numbers named. Demo restructured into functions with a `main()`. `pyproject.toml` pins lint/test config.

## E. Verification
* 55 → 63 tests; ruff clean at project config.
* Demo console output and `impact.json` byte-identical to pre-review; report evidence identical apart from the
  intentionally reworded Layer-1 note and +1 line numbers in `aer_leaky.py` (that file was reformatted).
* Mutation check: 15 realistic bugs injected one at a time. First run: 8 killed, 4 survived. Triage: 1 was dead code
  (removed), 3 were real test gaps (tests added). Re-run: **14/14 killed** (1 equivalent mutant excluded).
  Small, hand-picked sample — not exhaustive.
* Size: AST statements in the library 842 → 740 (−12%) while gaining class-method analysis. Raw non-blank lines went
  *up* (997 → 1,205) because the formatter expands one-liners and new behaviour was added — line count is not the metric.

## F. Not changed (honest list)
* `_call` is still radon grade D (24): a flat dispatcher, one guard clause per call shape. Also C-grade:
  `_check_event` (14), `visit_Assign` (14), `measure_impact` (13), `aggregate` (11).
* Layer 4 cost is dominated by 2 CV fits per feature plus a random forest; not optimised. MI z-threshold is still
  fragile with ~11 features.
* Layer 2 `register_split` and LSH bucketing are Python-level O(n) loops: fine at 10^4 rows, untested at 10^6.
  Decorating an estimator *class* patches it permanently; there is no un-patch.
* Layer 1 limits are unchanged (branches not modelled, unreferenced code invisible, unrecognised class names).
* Layer 3 is still validated only on synthetic data.


---
# Addendum — second-dataset pass (Titanic, Parkinson's speech, stocks)

Running on three datasets other than AER found problems the 63-test suite and the AER demo did not.

| # | Found by | Problem | Resolution |
|---|---|---|---|
| G1 | Titanic | `run_layer4` crashed on any NaN ("All the 5 fits failed") and could not handle string / high-cardinality columns. | `prepare_xy` now median-imputes with `<col>__missing` indicators, fills categorical NaN, drops ID-like / datetime / all-NaN columns, and returns a note for every step. Bad targets raise a clear `ValueError`. |
| G2 | pd_speech (252x750) | Layer 4 took **139 s**. KS permutation test 94 s (re-sorting each permutation), single-feature test 43 s (1,500 `cross_val_score` calls). | KS statistic vectorised and sorted once (verified equal to `scipy.ks_2samp`, including heavy ties); single-feature monotone AUC vectorised (identical values to the old logistic regression on AER). **14 s.** |
| G3 | pd_speech + noise control | "MI z >= 2 above the other features" flagged **42 of 750 pure-noise features** and 28 on pd_speech. On AER the true leaks only just cleared it (z = 2.10). | Replaced by an absolute rule: MI >= 50% of the target's entropy. 0 flags on noise; AER leaks now show 88%. Renamed `high-mutual-information`. |
| G4 | pd_speech | `perfect-separation` had the same multiple-comparisons flaw and mislabelled a clean tail of a legitimately strong feature (AUC 0.77) as "real-world features rarely give this". | Chance probability is Bonferroni-corrected over features x thresholds; severity 4 only if the rule classifies >= 90% of a class (or chance < 1e-12), otherwise severity 3 with a hedge. |
| G5 | stocks | Layer 3 reported `unmatched-value` ("matches no stored record") for rows that were simply absent from the training set. Ground truth: 0 unmatched values. | Absent rows are an information-level `labels-missing-from-training-set`; NULL joined values are `missing-joined-value`; only genuinely untraceable values are `unmatched-value`. |
| G6 | fresh venv | Plain `pytest` failed at collection (package not importable). | `pythonpath = ["."]` in `pyproject.toml`. |
| G7 | usability | Using the tool on a new dataset required writing Python. | `python -m leakdetect data.csv --target T --positive P` (Layer 4, optional `--script` for Layer 1; exit 1 on severity >= 4). |

Process notes: one of my own patches silently did nothing because `ruff format` had reshaped the text it expected; it was
caught by a failing test. Patches now assert their anchor. Mutation check on the new code: 12/12 killed after
strengthening four weak assertions (KS ties, median value, absent-row handling, CLI dtype mapping).
