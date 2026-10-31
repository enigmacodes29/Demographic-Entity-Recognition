# Checkpoints (not committed)

This folder is empty in the public repo on purpose — trained model weights are not made public.
`.gitignore` excludes everything here except this file and `.gitkeep`.

## What's needed, and by what

`04_qualitative_analysis.ipynb` loads saved checkpoints to introspect trained models (attention
weights, per-node votes, ablation/occlusion on a trained model rather than retraining). None of
the other notebooks (`01_baselines`, `02_graph_models_gat_rgcn`, `03_rgat`) need checkpoints —
they train fresh every run.

## How to get them

[TODO once this is finalized: wherever the actual checkpoint files are hosted for people who have
legitimate access — a private cloud storage link, an institutional drive, request-access
instructions, etc. Document the exact filenames expected and where to place them in this folder,
e.g.:]

```
checkpoints/
  gat_sbert_fold1.pt
  gat_bertiso_fold1.pt
  gat_bertctx_fold1.pt
  rgcn_sbert_fold1.pt
  ...
```

Single-fold weights only, as of now — see the main README for what that limits qualitative
analysis to (i.e. findings are illustrative of one fold's trained model, not averaged/validated
across folds or repeats).
