# BTAD PatchCore experiment

This folder contains the BTAD-only dataset adapter, runner, models, metrics and visual results.

The input dataset is read from:

```text
D:\patchcore\data\btad\BTech_Dataset_transformed
```

BTAD is handled as three independent product-specific PatchCore models:

```text
01: train/ok -> memory bank; test/ok + test/ko -> evaluation
02: train/ok -> memory bank; test/ok + test/ko -> evaluation
03: train/ok -> memory bank; test/ok + test/ko -> evaluation
```

Anomaly masks are matched by filename stem from `ground_truth/ko`.

Run all three categories with the same baseline configuration used for the local MVTec and VisA runs:

```powershell
$py = "C:\Users\xiaokun.wei\AppData\Local\miniconda3\envs\patchcore\python.exe"
& $py D:\patchcore\btadtest\run_btad_patchcore.py `
  --data-root D:\patchcore\data\btad `
  --output-root D:\patchcore\btadtest `
  --category all `
  --gpu 0 `
  --resize 256 `
  --imagesize 224 `
  --coreset 0.01
```

All generated BTAD experiment files are written below `D:\patchcore\btadtest`.
