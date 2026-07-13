# BTAD PatchCore Local Benchmark

- Data root: `D:\patchcore\data\btad`
- Categories: `01, 02, 03`
- Device: `cuda:0`
- Resize / image size: `256` / `224`
- Total elapsed seconds: `835.386`

| Category | Train | Test normal | Test anomaly | Image AUROC | Full pixel AUROC | Anomaly pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|
| btad_01 | 400 | 21 | 49 | 0.978620 | 0.966706 | 0.951384 |
| btad_02 | 399 | 30 | 200 | 0.816500 | 0.953429 | 0.945089 |
| btad_03 | 1000 | 400 | 41 | 0.999451 | 0.990796 | 0.923640 |

PRO is not included in this local benchmark.
