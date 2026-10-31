# [Paper title]

Reproducibility repo for [paper title] — DER-based stereotype/anti-stereotype classification on
InDiBias and IndiCasa, comparing non-neural baselines against R-GCN, GAT, and RGAT graph models,
with an inter-annotator agreement study and qualitative analysis.

## Repo layout

```
data/                     InDiBias/IndiCasa annotation JSONs
src/data/
  der_graph.py             DER annotation -> NetworkX graph (RuleBasedRelations etc.)
  featurizers.py            SBERT / BERT / BERT-ISO / BERT-CTX
  graph_convert.py           NetworkX -> PyTorch Geometric HeteroData
notebooks/
  00_llm_interannotator_agreement.ipynb   ChatGPT/Claude/Gemini DER-annotation agreement
  01_baselines.ipynb                      Majority/Identity-prior/LR variants/Sentence MLPs
  02_graph_models_gat_rgcn.ipynb          DER-only R-GCN (9 variants) + GAT (9 variants)
  03_rgat.ipynb                           DER-only RGAT, 3 encoders
  04_qualitative_analysis.ipynb           TODO — brings ablation/occlusion back
outputs/der_cross_annotator/   Inter-annotator agreement results (CSVs, per-model JSON annotations)
configs/                  Hyperparameters, eval protocol (TODO — currently inline in notebooks)
results/{tables,figures}/  Generated tables and figures land here
checkpoints/               Trained model weights — NOT public, gitignored, see checkpoints/README.md
```

## Setup

```bash
pip install -r requirements.txt   # TODO — not yet pinned
```

Needs: PyTorch, PyTorch Geometric, `sentence-transformers`, `transformers`, `scikit-learn`,
`networkx`, `pandas`. GPU recommended but not required.

## Running the notebooks

Run in this order — later notebooks depend on earlier ones' outputs:

1. **`00_llm_interannotator_agreement.ipynb`** — standalone, no dependency on the others. Validates
   annotation quality by having three LLMs independently annotate 50 sampled sentences.
2. **`01_baselines.ipynb`** — non-neural baselines and sentence-vector MLPs. Standalone.
3. **`02_graph_models_gat_rgcn.ipynb`** — R-GCN and GAT variants. Standalone (builds its own
   features; does not depend on 01 having run).
4. **`03_rgat.ipynb`** — RGAT variants. Standalone.
5. **`04_qualitative_analysis.ipynb`** *(not yet in repo)* — will depend on checkpoints from
   01-03.

Each of 01-03 loads `data/*.json`, builds DER graphs via `src/data/der_graph.py`, and evaluates
with repeated stratified 5-fold cross-validation (`SEED=42`, 3 repeats). Both datasets use plain
stratified CV (no minimal-pair grouping — the annotation files carry no template_id, so a grouped split offered no real benefit over plain stratified and was dropped).

## Checkpoints

Model weights are not committed — `checkpoints/` is gitignored except for a README explaining
what's expected there. See `checkpoints/README.md`.

## Known gaps

Model classes, the eval harness, and hyperparameters still live inline in each notebook rather
than in `src/`. No saved-fold file or formal test suite yet. `requirements.txt`, license, and
`CITATION.cff` don't exist yet.
