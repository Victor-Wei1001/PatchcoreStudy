# MVTec LOCO AD + PatchCore

This folder is self-contained for applying PatchCore to
`D:\patchcore\data\mvtec_loco_anomaly_detection`.

The corrected formal PatchCore benchmark uses `train/good` as the memory-bank
training set, `validation/good` to set an image-level threshold, and evaluates
only `test/good` plus `test/structural_anomalies`. Logical anomalies are not
included in the formal PatchCore metrics because they require global logical
and relational reasoning beyond ordinary local patch similarity.
Logical anomaly masks are merged from all mask components in the corresponding
`ground_truth/logical_anomalies/<image_id>` directory.

Run from PowerShell with the PatchCore conda environment:

```powershell
C:\Users\xiaokun.wei\AppData\Local\miniconda3\envs\patchcore\python.exe `
  D:\patchcore\mvteclocotest\run_mvtec_loco_patchcore.py
```

Each run creates a timestamped `benchmark_mvtec_loco_structural_*` directory containing
the memory-bank models, predictions, visualizations, CSV/JSON metrics, and a
text report. The report records training/validation/test-inference time, the
validation-derived threshold and test classification statistics, weighted
inference time per image, `nvidia-smi` average/peak GPU utilization and VRAM,
and CUDA allocator peak allocated/reserved memory.
