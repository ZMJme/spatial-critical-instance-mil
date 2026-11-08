# Deliberately excluded from this repository

- Raw `database/` contents, WSI/pathology images, patches and annotation files: dataset redistribution terms and volume.
- All `.pth/.pt/.pkl` features and checkpoints: large binary artifacts, dataset-derived content and unresolved cached encoder provenance.
- `BRACS.xlsx` and other original patient-level metadata: not necessary for the code release; original access conditions apply.
- Notebook outputs, embedded images, execution counts and incidental notebook metadata: removed from packaged copies only.
- `.git`, credentials, SSH keys, environments, native DLLs, caches, notebook checkpoints and archives: not source deliverables.
- Historical duplicate source trees, legacy experiment directories and unselected runs: avoid mixing paper results with abandoned workflows. Audited main runs, new ablations and labeled RNN diagnostics are retained as JSON only.
- Manuscript drafts/PDFs, reviewer notes, author/journal documents and reference-paper PDFs: outside this code/data release.
- Raw training logs: omitted; structured configurations and summaries are retained where available.

The full local exclusion inventory is supplied OUTSIDE the upload ZIP. Excluded files are not deleted from the workspace.
Dataset/weights release can be handled separately through an appropriate repository after access and provenance are established. No GitHub upload or repository publication is performed by this packaging step.
