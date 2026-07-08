# PatchCore Study

This repository contains five defect datasets exploration code and generated results.

## Current Work



The current scripts focus on:

- loading  MVTec AD, VisA, BTAD, KolektorSDD2, and MVTec LOCO AD datasets
- visualizing original images, ground-truth masks, overlays, and bounding boxes
- calculating defect area from binary ground-truth masks
- generating category-wise and sub-category-wise defect size reports
- generating visualization figures
- generating comparison charts for the selected five datasets

## Defect Size Calculation

For each defective image:

defect area ratio = number of non-zero pixels in the ground-truth mask / full image area

The reports include:

- per-image defect size report
- category-wise average defect size report
- sub-category-wise average defect size report
- visualization figures

## Note

The original dataset is not included because it is large. Only scripts and generated analysis outputs are uploaded.