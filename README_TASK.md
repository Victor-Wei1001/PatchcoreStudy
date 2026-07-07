# PatchCore defect-dataset task

This workspace is organized as:

- `patchcore-inspection-main/`: Amazon Science PatchCore implementation.
- `tools/`: dataset exploration, visualization, and benchmark wrappers.
- `ui/`: PyQt6 demo app for loading an image and showing defect overlays.
- `data/`: put downloaded datasets here.
- `outputs/`: generated reports, figures, models, and logs.

## Required local setup

This machine currently does not have a usable Python interpreter in PATH. Install Python 3.10 or Miniconda first. PatchCore's README was written for Python 3.8, but Python 3.10 is usually easier on current Windows.

Recommended:

```powershell
winget install -e --id Anaconda.Miniconda3
```

Then open a new PowerShell:

```powershell
conda create -n patchcore python=3.10 -y
conda activate patchcore
cd D:\patchcore\patchcore-inspection-main
pip install -r requirements.txt
pip install psutil pandas opencv-python pyqt6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If CUDA PyTorch install fails, use CPU PyTorch:

```powershell
pip install torch torchvision
```

## Dataset layout

Put MVTec AD under:

```text
D:\patchcore\data\mvtec
```

Expected example:

```text
D:\patchcore\data\mvtec\bottle\train\good\*.png
D:\patchcore\data\mvtec\bottle\test\broken_large\*.png
D:\patchcore\data\mvtec\bottle\ground_truth\broken_large\*_mask.png
```

Amazon's README lists 15 MVTec AD categories:

`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

## Explore data

```powershell
conda activate patchcore
cd D:\patchcore
python tools\dataset_report.py --root data\mvtec --dataset mvtec --out outputs\mvtec_report.csv
python tools\visualize_dataset.py --root data\mvtec --dataset mvtec --categories bottle cable --rows 5 --out outputs\bottle_vs_cable.png
python tools\visualize_masks.py --root data\mvtec --category bottle --rows 5 --out outputs\bottle_masks.png
```

## Train one category first

Start with one category on your 8GB GPU:

```powershell
conda activate patchcore
cd D:\patchcore
python tools\run_patchcore_benchmark.py --category bottle --data-root data\mvtec --gpu 0 --resize 256 --imagesize 224 --coreset 0.01
```

This writes a JSON report in `outputs\benchmarks`.

For all MVTec categories, repeat with `--category all`, but expect long runtime.

## PyQt6 demo

After training a model:

```powershell
conda activate patchcore
cd D:\patchcore
python ui\patchcore_defect_ui.py
```

The UI is intentionally small: it loads an image, an optional ground-truth mask, and shows original, mask overlay, predicted heatmap placeholder, and side-by-side comparison. Connecting the Amazon PatchCore model loader is left as the final integration step once a saved model path exists.
