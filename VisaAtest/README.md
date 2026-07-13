# VisA PatchCore reproduction

This folder contains the VisA-only runner. It does not modify the original
PatchCore source or benchmark scripts.

The runner reads the official `data/VisA/split_csv/1cls.csv` split. For each
VisA category it builds a separate memory bank from `train + normal` images,
then evaluates on that category's `test + normal` and `test + anomaly` images.

Run all 12 categories with the PatchCore settings used by the local MVTec
benchmark:

```powershell
$py = "C:\Users\xiaokun.wei\AppData\Local\miniconda3\envs\patchcore\python.exe"
& $py D:\patchcore\VisaAtest\run_visa_patchcore.py `
  --data-root D:\patchcore\data\VisA `
  --output-root D:\patchcore\VisaAtest `
  --category all `
  --gpu 0 `
  --resize 256 `
  --imagesize 224 `
  --coreset 0.01
```

The output folder contains per-category PatchCore models, selected visual
results, `results.csv`, and a Markdown report. The reported metrics are image
AUROC, full pixel AUROC, and anomaly-only pixel AUROC. PRO is not included in
the current local MVTec benchmark implementation.
