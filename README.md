# Spatial context and critical-instance aggregation for WSI classification

Research code and archived experiment records for CAMELYON16 and a four-class BRACS subset.
This is a curated snapshot of the original notebook workflow, not a standalone, fully portable training library.

## Contents
- `WSI_pretrain/`: output-cleared notebooks, original training runners and `wsi` modules.
- `splits/`: exact archived training/validation file lists (CAMELYON16 216/54; BRACS 99/25).
- `results/`: audited main comparisons, CAMELYON16 ablations and explicitly retained historical RNN diagnostics. Directory names distinguish these runs; do not pool duplicate reruns.
- `docs/`: data preparation, reproduction boundaries and exclusion policy.
- `MANIFEST.json`: source-to-package mapping and SHA-256 of packaged copies.

## Environment
`requirements.in` lists inferred dependencies, not pinned versions from the original GPU environment.
Install a mutually compatible PyTorch/torchvision/CUDA/PyG stack. OpenSlide also requires its native library.
The exact historical environment has not been recovered or tested here. Source syntax and JSON structure are checked during packaging; training has not been rerun.

## Original notebook workflow
1. Obtain data through the official dataset access procedures; see `docs/DATA.md`.
2. Open `WSI_pretrain/MIL_compare.ipynb` and execute the import, data loader and required model-definition cells. Do not run all training cells blindly: the notebook contains historical and alternative workflows.
3. Put the corresponding archived split from `splits/` in your chosen save root BEFORE training. The original runner otherwise creates a new split. Keep the exact slide membership and order.
4. Update local data paths, checkpoint paths and output paths in notebook cells. Archived JSON paths describe the original `/data/...` environment and were deliberately preserved. The runners verify the split's `data_path`; update this field only to your local equivalent, without changing its filename lists.
5. From that notebook, run (example):
```python
%run -i retrain_baselines_patience10_on_CAMELYON16.py --dataset CAMELYON16 --models Ours --data-path /path/to/training/20xbags --save-root /path/to/new/results
```
For BRACS use `retrain_baselines_patience10_on_BRACS.py`, `--dataset BRACS` and its data/split paths. These scripts depend on the notebook namespace, including model classes and `get_data`; they are not independent CLI programs.
6. The original ablation code lives in `ablation.ipynb`; restore its dependencies and use the mapped A0 slide identities for lower-resolution bags. Evaluation cells are included in notebooks. Keep final testing separate from model selection.

## Interpretation
Ours uses equal-weight bag CE and max-instance CE. Baseline objectives remain model-specific; not every baseline uses joint CE.
The archived stopping implementation resets to 10 on improvement and decrements in the same epoch, yielding nine subsequent non-improving epochs; do not silently replace it when reproducing these results.
The reported runs are single runs with no explicit training seed. No cross-seed or significance claim is implied.
MIL-RNN Top10 and the historical all-instance RNN are different workflows and must be distinguished.
BRACS includes patient overlap across slide partitions; these records are not patient-independent evaluation.
Cached DINO checkpoint/pretraining lineage is not fully established. Released source does not establish that lineage.

## Rights and release status
Raw datasets, features, weights, patient metadata spreadsheets, manuscript drafts and notebook outputs are NOT included.
See `docs/EXCLUDED.md`. No new open-source license has been assigned: project ownership and any upstream code notices need author review before granting a license. Existing source notices are preserved. This package does not claim that all baseline implementations are official reproductions.
No synthetic or fabricated example data are presented as experimental observations.
