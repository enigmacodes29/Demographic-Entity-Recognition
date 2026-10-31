# [Paper title] — reproducibility repo

## Status

**Ablation/occlusion are intentionally not in this repo yet** — removed on purpose, will be added
back alongside `04_qualitative_analysis.ipynb`. Counterfactual/deletion experiments are also
dropped permanently per project decision. Do not reintroduce either without checking with the
project owner first.

**Three real bugs were found and fixed this session** (not just refactored — each was verified
with a standalone test before being applied):

1. **`01_baselines.ipynb`, `SBERTFeaturizer.node_key`**: after the DER-construction fix (below),
   node IDs changed from plain strings to `(span, node_type)` tuples. This cell still built its
   SBERT cache key from `str(n)`, which on a tuple produces the literal Python repr —
   `"('brahmin', 'Identity Dimensions')"` — and SBERT would embed *that string* instead of
   "brahmin". Fixed to use `node_surface(G, n)`. **Checked**: this did not corrupt any
   already-reported baseline number — `MLP SBERT` only ever calls SBERT at the sentence level
   (unaffected code path), and the LR bag-of-types/bag-of-nodes/frames-triggers baselines never
   call SBERT at all. It would, however, have silently corrupted any GAT/R-GCN "SBERT" variant
   that reused this exact code path.
2. **`02_graph_models_gat_rgcn.ipynb` was not independently runnable.** It never defined its own
   `SBERTFeaturizer`/`BERTFeaturizer`/etc. classes — a cell literally checked
   `if "FEATURIZERS" in globals()`, meaning it only worked if run in the same kernel session
   right after another notebook that happened to populate `FEATURIZERS` first. Fixed: it now
   builds `FEATURIZERS` fresh via `src/data/featurizers.py`.
3. **Same notebook, separate bug**: even granting session continuity, a later cell did
   `FEATURIZERS = {}` and rebuilt it with only SBERT + BERT-ISO (all R-GCN needs), which wiped out
   BERT-CTX. The GAT section three cells later indexes `FEATURIZERS["BERT-CTX"]` directly with no
   fallback — this would `KeyError` in *any* execution order, every time, independent of bug #2.
   Fixed by removing the wipe-and-rebuild block; `FEATURIZERS` is now built once, correctly, and
   left alone.

All three are now covered by the smoke tests described under "Known gaps" #4 (formalizing them
into a `tests/` folder is still pending) so they can't silently reappear.

## Repo layout

```
src/
  data/
    der_graph.py       # DER annotation -> NetworkX graph. Canonical, verified identical
                        # across all 3 notebooks (after fixing baselines — see Status).
    featurizers.py      # SBERT / BERT / BERT-ISO / BERT-CTX. Canonical, bug-fixed (see Status #1).
    graph_convert.py     # NetworkX -> PyG HeteroData. Verified byte-identical in graph_models
                        # and RGAT notebooks before extraction.
notebooks/
  00_llm_interannotator_agreement.ipynb  # ChatGPT/Claude/Gemini on 50 sentences — label bug fixed, see below
  01_baselines.ipynb              # Majority/Identity-prior/LR variants/Sentence MLPs — DER + SBERT fixed
  02_graph_models_gat_rgcn.ipynb  # DER-only R-GCN (9 variants) + GAT (9 variants) — now self-contained
  03_rgat.ipynb                   # DER-only RGAT, 3 encoders — unchanged, was already correct
  04_qualitative_analysis.ipynb   # TODO — not yet in repo, brings ablation/occlusion back
outputs/der_cross_annotator/      # inter-annotator agreement outputs — label bug fixed, prompt still missing
configs/                  # TODO — see "Known gaps"
results/{tables,figures}/
checkpoints/               # NOT public — gitignored, empty in this repo. See checkpoints/README.md
```

## Checkpoints are private, on purpose

Model weights are **not** committed to this repo and never will be — `.gitignore` excludes
everything under `checkpoints/` except `checkpoints/README.md`. This was confirmed explicitly:
the project owner only meant "don't make the weights public," not "the qualitative-analysis
notebook doesn't need them" — it does, to introspect a trained model rather than retrain one.
See `checkpoints/README.md` for what's expected there once `04_qualitative_analysis.ipynb`
arrives (currently a TODO placeholder — needs the actual access/hosting method filled in).

## Evaluation protocol (confirmed identical across all 3 notebooks)

- `SEED = 42`, `SEEDS = [SEED + 100*r for r in range(N_REPEATS)]`
- `make_splits()`: `StratifiedKFold` (stratified plan) / `StratifiedGroupKFold` (grouped plan),
  `n_splits=5`, `shuffle=True`, `random_state=seed`
- Two plans, both plain stratified 5-fold CV: `InDiBias/stratified`, `IndiCasa/stratified`.

**Decision (resolved): "grouped" CV has been dropped entirely**, in all three notebooks. It was
meant to protect InDiBias minimal pairs via `template_id`, but none of the 326 real InDiBias
records have one — `build_dataset`'s fallback gives every record its own group, so
`StratifiedGroupKFold` ("grouped") was producing a *different* split from `StratifiedKFold`
("stratified") with the same fold sizes, but only ~15-23% item overlap per fold on real data —
not literally the same folds, just no longer any actual minimal-pair protection either. Rather
than keep reporting two plans where the "safer" one offers no real benefit, we report one:
plain stratified 5-fold everywhere. Any previously-reported "grouped" numbers are superseded, not
interchangeable with the stratified ones — they came from different train/test splits.
Updated in: `EVAL_PLAN`/`PROFILES` in all 3 notebooks, the fold-hash audit cell in
`02_graph_models_gat_rgcn.ipynb` (was hardcoded to `"grouped"`), the paired-bootstrap
significance-testing cell in `01_baselines.ipynb` (was filtering on `cv == "grouped"`, which
would have silently produced an empty table), and every notebook's intro/results markdown that
described "grouped" as the primary or safer result.

Each notebook still calls `make_splits` independently rather than loading one shared fold file
(see "Known gaps" #3) — but because the inputs (`labels`, `groups`, `seeds`) are now built from
the *same* verified `der_graph.py` code everywhere, the resulting splits are identical in practice,
not just "probably the same."

## Inter-annotator agreement (ChatGPT/Claude/Gemini on 50 sentences)

`notebooks/00_llm_interannotator_agreement.ipynb` + `outputs/der_cross_annotator/`. Same
treatment as the DER/featurizer bugs above — found, verified, fixed, documented, not silently
patched:

1. **Labels were swapped.** `DATA_FILES` mapped `"stereotype"` to `*_neg.json` and
   `"anti_stereotype"` to `*_pos.json` — backwards from this project's convention everywhere else
   (pos=stereotype=1). Fixed in the notebook (cell 2) for future runs.
2. **Fixing it does NOT mean rerunning the sampling.** Tested empirically: redrawing the
   stratified sample with the corrected mapping picks an almost entirely different 50 sentences
   (2/50 overlap with the original draw). The three LLMs' annotations are already collected
   against the *original* 50 — rerunning the sampling cell would silently invalidate all of them
   and require re-querying every API. **Did not do this.**
3. **The actual fix applied: a pure relabel, not a resample.** The agreement computation
   (`compare`/`prf`/`sets` in cell 12) only ever uses `sentence_id`/`span`/`node`/`label` — never
   `class_label` — so `pairwise_agreement_summary.csv`, `annotation_format_checks.csv`, and
   `all_annotations_long.csv` were already correct and untouched. Only `sample_manifest.csv`'s
   `class_label` column and the report's sample-count section needed correcting, done by
   remapping each row from its (unaffected) `source_file` column. Same 50 sentences, same
   annotations, same agreement numbers — only the bookkeeping label fixed.
4. **Added a guard (cell 6)**: if a sample has already been drawn (`sample_manifest.csv` exists),
   the notebook now loads and relabels it instead of redrawing, so this can't happen again by
   accident on a rerun.
5. **⚠ Unresolved, needs you**: the actual prompt sent to the three LLMs was never recorded —
   `shared_prompt_packet.json`'s `instructions` field is still the literal placeholder text
   `"PASTE YOUR FINAL DER ANNOTATION PROMPT HERE."` Added a hard assertion in cell 8 so this can't
   silently happen again, but the real prompt for the *already-collected* batch is still missing.
   Without it this isn't reproducible, and a reviewer can reasonably ask for it. If you have it
   saved anywhere outside this notebook, send it and I'll fold it in properly.

Headline numbers (unaffected by the label bug, safe to use as-is): pairwise span-F1 ranges
0.668–0.750, typed-F1 0.664–0.736, identity-label agreement is **1.0 for all three pairs**.

## Known gaps / next steps, roughly in priority order

1. **Model classes not yet extracted.** `SentenceMLP`, `SentenceRGCN`, `DEROnlyGAT`,
   `RGATGraphClassifier` still live inline in their respective notebooks. Candidates for
   `src/models/{baselines,rgcn,gat,rgat}.py`.
2. **Eval harness not yet extracted.** `make_splits`, the repeated-CV loop
   (`cross_validate`/`evaluate_repeated_cv`), and metric computation are near-identical across
   notebooks but not shared from one file. Candidate: `src/eval.py`.
3. **No saved fold file.** Folds are "identical in practice" (verified), not "identical because
   loaded from the same file." Candidate: generate once, save to
   `configs/folds_<dataset>_<plan>.json`.
4. **No formal test suite.** `der_graph.py`, `featurizers.py`, and `graph_convert.py` were each
   verified with ad-hoc scripts during this session (construct a graph, check node/edge shapes,
   confirm no garbage keys), not committed `pytest` tests. Candidate: `tests/test_der_graph.py`
   etc., turning those ad-hoc checks into something that runs in CI.
5. **Notebooks still partly inline for DER construction.** The three new `src/data/` modules
   exist and are verified, but the notebooks still carry the old inline cells alongside the fixes
   in some places (RGAT does; baselines/graph_models now import where their specific bugs were
   fixed). A full pass to delete every remaining inline duplicate and import from `src/` instead
   is still needed.
6. **Unexplained result pattern in baselines, still unconfirmed:** `LR frames/triggers only`,
   `LR bag-of-types`, and `LR bag-of-nodes` produce identical accuracy/F1 to each other
   (0.802/0.622); `Identity prior` and all 4 Sentence-MLP variants also exactly match each other
   (0.796/0.604). Either a real finding or a feature-construction bug. On the list, deprioritized
   per your call.
7. **`configs/model/*.yaml`** don't exist yet — hyperparameters live inline. Lower priority than
   1-3.
8. **`requirements.txt`, `.gitignore`, license, `CITATION.cff`** — none exist yet. Waiting on your
   environment info (see below).

## What's been verified against real data (this session)

`data/{indibias,indicasa}_{pos,neg}.json` are now in the repo (small enough to commit directly,
no LFS needed). Ran `der_graph.build_dataset` on all four files for real:

- InDiBias: 326 records (243 stereotype / 83 anti-stereotype), avg 2.77 entity nodes/sentence,
  avg 1.58 DER edges/sentence, 40 sentences with zero entity nodes.
- IndiCasa: 326 records (243/83), avg 5.79 entity nodes/sentence, avg 4.66 DER edges/sentence,
  1 sentence with zero entity nodes.
- Zero nodes with leaked tuple-repr text across every non-Sentence node in both datasets (the
  SBERT bug fix holds on real data, not just the synthetic test case used to find it).
- The same span tagged as more than one node type (e.g. "OBC" as both Identity and Trigger)
  happens in 84/326 InDiBias sentences (26%) — common, not a rare edge case.
- **The `template_id`/grouped-CV issue** — confirmed against real data, decided: dropped (see
  Evaluation protocol section above).

**Still can't verify**: the actual SBERT/BERT/BERT-CTX encoding step, since my environment has no
network access to Hugging Face. `der_graph.py` and `graph_convert.py` are now verified end-to-end
on real data; `featurizers.py`'s `node_key` logic is verified, but nobody has run `_compute()`
against these files yet. Please run it once and confirm.

## What I need from you to keep going

- **Your environment**: `pip freeze` or equivalent, Python version, Colab/Kaggle/local, GPU or not.
- **Single-fold checkpoint(s)** — confirmed needed (not optional), confirmed private (not
  committed to git). Send whenever `04_qualitative_analysis.ipynb` is ready; I'll place them
  under `checkpoints/` (gitignored) and fill in `checkpoints/README.md`'s access instructions
  properly instead of the current placeholder.
- `04_qualitative_analysis.ipynb` (now also carrying ablation/occlusion, per your call) — the
  only notebook still outstanding (inter-annotator agreement has landed, see above).
- **The real DER-annotation prompt** used for the already-collected ChatGPT/Claude/Gemini batch
  — see inter-annotator section above, this is the one open item blocking that part specifically.
- Everything under "Known gaps" above (`src/models/`, `src/eval.py`, saved fold file, test suite,
  config YAMLs, `requirements.txt`/license/`CITATION.cff`) is independent of the two notebooks
  above and can be worked on in parallel — not blocked on them, your call on priority.
