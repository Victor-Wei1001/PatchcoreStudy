# VisA PatchCore Local Benchmark

- Data root: `D:\patchcore\data\VisA`
- Split CSV: `D:\patchcore\data\VisA\split_csv\1cls.csv`
- Categories: `candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum`
- Device: `cuda:0`
- Total elapsed seconds: `1208.647`

| Category | Train | Test normal | Test anomaly | Image AUROC | Full pixel AUROC | Anomaly pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|
| visa_candle | 900 | 100 | 100 | 0.987900 | 0.989288 | 0.981923 |
| visa_capsules | 542 | 60 | 100 | 0.748500 | 0.984697 | 0.983489 |
| visa_cashew | 450 | 50 | 100 | 0.974200 | 0.982445 | 0.976416 |
| visa_chewinggum | 453 | 50 | 100 | 0.987800 | 0.984316 | 0.978231 |
| visa_fryum | 450 | 50 | 100 | 0.960400 | 0.905978 | 0.887374 |
| visa_macaroni1 | 900 | 100 | 100 | 0.971000 | 0.993687 | 0.989953 |
| visa_macaroni2 | 900 | 100 | 100 | 0.770200 | 0.976544 | 0.973086 |
| visa_pcb1 | 904 | 100 | 100 | 0.987600 | 0.996375 | 0.994048 |
| visa_pcb2 | 901 | 100 | 100 | 0.971700 | 0.982767 | 0.970025 |
| visa_pcb3 | 905 | 101 | 100 | 0.984356 | 0.990130 | 0.982030 |
| visa_pcb4 | 904 | 101 | 100 | 0.996634 | 0.974945 | 0.953363 |
| visa_pipe_fryum | 450 | 50 | 100 | 0.999000 | 0.988771 | 0.983748 |

PRO is not included; this matches the existing local MVTec benchmark metrics.
