```markdown
# PatchCore Study

This repository contains five defect datasets exploration code , generated results and local running records.

## Current Work

The current scripts focus on:

- loading MVTec AD, VisA, BTAD, KolektorSDD2, and MVTec LOCO AD datasets
- visualizing original images, ground-truth masks, overlays, and bounding boxes
- calculating defect area from binary ground-truth masks
- generating category-wise and sub-category-wise defect size reports
- generating visualization figures
- generating comparison charts for the selected five datasets
- running the official PatchCore code locally with GPU support
- recording PatchCore running statistics and detection visualizations

## Dataset Report

For each defective image:

defect area ratio = number of non-zero pixels in the ground-truth mask / full image area

The reports include:

- per-image defect size report
- category-wise average defect size report
- sub-category-wise average defect size report
- visualization figures

## PatchCore Local Running Report

A local PatchCore experiment was conducted using the official open-source implementation.

The report includes:

- training time and GPU memory usage
- inference time and inference time per image
- trained model output size
- anomaly detection visualizations compared with ground-truth masks

The first local test was performed on the MVTec AD `bottle` category using GPU.

## Note

The original dataset is not included because it is large. Only scripts and generated analysis outputs are uploaded.
```
