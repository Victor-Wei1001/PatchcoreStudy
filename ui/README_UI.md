# PatchCore PyQt6 UI

This folder contains the desktop UI for running the locally trained PatchCore models on a single image.

## Supported datasets

- MVTec AD: models are loaded from the existing MVTec benchmark output.
- VisA: models are loaded from `D:\patchcore\VisaAtest\benchmark_visa_20260713_110839\models`.

Each category uses its own model and memory bank. The UI supports automatic category detection from paths under `data\mvtecAD` or `data\VisA`.

## Start UI

```powershell
conda activate patchcore
cd D:\patchcore
python ui\patchcore_defect_ui.py
```

If the environment has a fixed Python path:

```powershell
& "C:\Users\xiaokun.wei\AppData\Local\miniconda3\envs\patchcore\python.exe" D:\patchcore\ui\patchcore_defect_ui.py
```

## Usage

1. Select `MVTec AD` or `VisA`.
2. Select the matching category, or load an image from the corresponding dataset folder and let the UI detect it automatically.
3. Adjust `Resize` if needed. The trained models still receive a `224 x 224` center crop.
4. Adjust the raw `Decision threshold` if needed. The saved PatchCore models do not contain a calibrated classification threshold; the default is `1.5`.
5. Load an image and click `Run Detection`.

For VisA, masks use low grayscale label values rather than necessarily using 255 for foreground. The UI therefore treats every mask pixel greater than zero as defect foreground and uses nearest-neighbor interpolation for mask resizing.

## Result information

The result card shows:

- Prediction: normal or anomaly according to the configurable raw-score threshold.
- Defect score: the raw PatchCore image score.
- Prediction map maximum: the maximum raw segmentation score.
- Inference time: model prediction time only.
- Total time: model loading, preprocessing and prediction time.
- Resize and model input size.
- Device and category-specific model.

The UI keeps up to two recently used models in memory, so repeated inference with the same category avoids reloading the FAISS index and backbone.

Only files in this `ui` folder are part of the UI implementation. The dataset and trained model folders are read-only inputs for the application.
