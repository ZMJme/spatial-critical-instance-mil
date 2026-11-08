# Data access and schema

Obtain CAMELYON16 from its official project resources: https://camelyon16.grand-challenge.org/
Obtain BRACS through its official access process: https://www.bracs.icar.cnr.it/
Do not redistribute either dataset or derived cached features merely because the data were available for this study. Follow the applicable dataset terms.

Expected archived feature files are `.pth` dictionaries containing `features` (N x 384), `label`, `rects` (N coordinates) and `patch_size`; scale metadata may also contain `level` and `level_base`.
The archived loader constructs eight-neighbor spatial edges from coordinates and patch size. Preserve alignment of feature rows and coordinates.
CAMELYON16 uses training/testing directories with `20xbags`, `10xbags`, `5xbags`. BRACS runs use the study's existing four-class feature subset.
Class names/mapping must be established from the preprocessing and authoritative metadata; do not infer them from integer labels or folder names. The original metadata spreadsheet is deliberately omitted.

Only study-generated split manifests, run configuration, metrics and slide-level prediction records are included here. They contain public slide identifiers, not the patient metadata spreadsheet. The archive preserves the original local paths as provenance; replace paths locally without redrawing splits.
There are no downloadable feature files or pretrained weights in this package. Reproducing inference requires obtaining/producing those separately; the cached encoder lineage remains a documented limitation.
