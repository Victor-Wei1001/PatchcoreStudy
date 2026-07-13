# VisA PatchCore Local Benchmark

- Data root: `D:\patchcore\data\VisA`
- Split CSV: `D:\patchcore\data\VisA\split_csv\1cls.csv`
- Categories: `candle`
- Device: `cuda:0`
- Total elapsed seconds: `145.332`

| Category | Train | Test normal | Test anomaly | Image AUROC | Full pixel AUROC | Anomaly pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|
| visa_candle | 900 | 100 | 100 | 0.987900 | 0.989288 | 0.981923 |

PRO is not included; this matches the existing local MVTec benchmark metrics.
