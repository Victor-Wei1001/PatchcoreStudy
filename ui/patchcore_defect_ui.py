import os
import ssl
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except Exception:
    pass

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    print("PyQt6 is not installed in this environment.")
    print("Install it with:")
    print(
        r"  C:\Users\xiaokun.wei\AppData\Local\miniconda3\envs\patchcore\python.exe -m pip install PyQt6"
    )
    raise exc

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PATCHCORE_ROOT = Path(r"D:\patchcore")
PATCHCORE_SRC = PATCHCORE_ROOT / "patchcore-inspection-main" / "src"
MVTec_MODEL_ROOT = (
    PATCHCORE_ROOT
    / "outputs"
    / "patchcore_runs"
    / "PatchCoreLocal"
    / "benchmark_all_20260708_110050"
    / "models"
)
VISA_MODEL_ROOT = (
    PATCHCORE_ROOT
    / "VisaAtest"
    / "benchmark_visa_20260713_110839"
    / "models"
)

if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))

import patchcore.common
import patchcore.patchcore


MVTec_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

DATASET_CONFIG = {
    "MVTec AD": {
        "classes": MVTec_CLASSES,
        "model_root": MVTec_MODEL_ROOT,
        "model_prefix": "mvtec_",
        "default_folder": PATCHCORE_ROOT / "data" / "mvtecAD",
        "default_resize": 256,
    },
    "VisA": {
        "classes": VISA_CLASSES,
        "model_root": VISA_MODEL_ROOT,
        "model_prefix": "visa_",
        "default_folder": PATCHCORE_ROOT / "data" / "VisA",
        "default_resize": 256,
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_INPUT_SIZE = 224
MODEL_CACHE_LIMIT = 2
MODEL_CACHE = OrderedDict()


def pil_to_pixmap(image):
    image = image.convert("RGB")
    arr = np.ascontiguousarray(np.array(image))
    height, width, channels = arr.shape
    qimage = QImage(
        arr.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


def normalize_map(mask):
    mask = np.asarray(mask, dtype=np.float32)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(mask.min())
    max_value = float(mask.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(mask, dtype=np.float32)
    return (mask - min_value) / (max_value - min_value)


def heatmap_from_mask(mask):
    normalized = normalize_map(mask)
    if float(normalized.max()) <= 0.0:
        rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    else:
        rgba = cm.get_cmap("jet")(normalized)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def overlay_heatmap(image, heatmap, alpha=0.45):
    return Image.blend(image.convert("RGB"), heatmap.convert("RGB"), alpha)


def preprocess_image(image_path, resize):
    display_transform = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    display_image = display_transform(image)
    tensor = tensor_transform(image).unsqueeze(0)
    return display_image, tensor


def preprocess_mask(mask_path, resize):
    if not mask_path:
        return None
    mask_transform = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
        ]
    )
    mask = Image.open(mask_path).convert("L")
    # VisA uses low-valued grayscale labels (1..8), not necessarily 255.
    mask = mask.point(lambda value: 255 if value > 0 else 0)
    return mask_transform(mask)


def dataset_config(dataset_name):
    return DATASET_CONFIG[dataset_name]


def resolve_model_root(dataset_name):
    config = dataset_config(dataset_name)
    configured = config["model_root"]
    if configured.exists():
        return configured

    if dataset_name == "VisA":
        candidates = sorted(
            (PATCHCORE_ROOT / "VisaAtest").glob("benchmark_visa_*/models"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    else:
        candidates = sorted(
            (PATCHCORE_ROOT / "outputs" / "patchcore_runs" / "PatchCoreLocal").glob(
                "benchmark_all_*/models"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    return candidates[0] if candidates else configured


def model_path_for(dataset_name, category):
    config = dataset_config(dataset_name)
    return resolve_model_root(dataset_name) / f"{config['model_prefix']}{category}"


def infer_dataset_from_path(image_path):
    parts = [part.lower() for part in Path(image_path).parts]
    if "visa" in parts:
        return "VisA"
    if "mvtecad" in parts:
        return "MVTec AD"
    return None


def infer_category_from_path(image_path, dataset_name):
    parts = list(Path(image_path).parts)
    anchor = "VisA" if dataset_name == "VisA" else "mvtecAD"
    for index, part in enumerate(parts):
        if part.lower() == anchor.lower() and index + 1 < len(parts):
            category = parts[index + 1]
            if category in dataset_config(dataset_name)["classes"]:
                return category
    return None


def find_mvtec_mask(image_path):
    path = Path(image_path)
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "test" not in lowered:
        return ""
    test_index = lowered.index("test")
    if test_index + 1 >= len(parts):
        return ""
    defect_type = parts[test_index + 1]
    if defect_type.lower() == "good":
        return ""
    category_dir = Path(*parts[:test_index])
    mask_path = category_dir / "ground_truth" / defect_type / f"{path.stem}_mask{path.suffix}"
    return str(mask_path) if mask_path.exists() else ""


def find_visa_mask(image_path):
    path = Path(image_path)
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "visa" not in lowered or "images" not in lowered:
        return ""
    images_index = lowered.index("images")
    if images_index + 1 >= len(parts):
        return ""
    split_name = parts[images_index + 1].lower()
    if split_name == "normal":
        return ""
    # The parent immediately before Images is Data; move one level higher to
    # the VisA category directory before appending Data/Masks/Anomaly.
    category_dir = Path(*parts[: images_index - 1])
    mask_path = category_dir / "Data" / "Masks" / "Anomaly" / f"{path.stem}.png"
    return str(mask_path) if mask_path.exists() else ""


def find_ground_truth_mask(dataset_name, image_path):
    if dataset_name == "VisA":
        return find_visa_mask(image_path)
    return find_mvtec_mask(image_path)


def load_cached_model(model_path, device):
    cache_key = (str(model_path), str(device))
    if cache_key in MODEL_CACHE:
        MODEL_CACHE.move_to_end(cache_key)
        return MODEL_CACHE[cache_key], True

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=4)
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(str(model_path), device=device, nn_method=nn_method)
    MODEL_CACHE[cache_key] = model
    MODEL_CACHE.move_to_end(cache_key)

    while len(MODEL_CACHE) > MODEL_CACHE_LIMIT:
        _, old_model = MODEL_CACHE.popitem(last=False)
        del old_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model, False


class InferenceWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, dataset_name, category, image_path, mask_path, resize, threshold):
        super().__init__()
        self.dataset_name = dataset_name
        self.category = category
        self.image_path = image_path
        self.mask_path = mask_path
        self.resize = resize
        self.threshold = threshold

    def run(self):
        try:
            total_start = time.perf_counter()
            model_path = model_path_for(self.dataset_name, self.category)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model_start = time.perf_counter()
            model, cache_hit = load_cached_model(model_path, device)
            model_load_time = time.perf_counter() - model_start

            display_image, tensor = preprocess_image(self.image_path, self.resize)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_start = time.perf_counter()
            score_list, mask_list = model.predict(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - inference_start

            raw_score = float(np.asarray(score_list[0]).reshape(-1)[0])
            raw_map = np.asarray(mask_list[0], dtype=np.float32)
            heatmap = heatmap_from_mask(raw_map)
            overlay = overlay_heatmap(display_image, heatmap)
            gt_mask = preprocess_mask(self.mask_path, self.resize)
            total_time = time.perf_counter() - total_start

            self.finished.emit(
                {
                    "dataset": self.dataset_name,
                    "category": self.category,
                    "score": raw_score,
                    "threshold": self.threshold,
                    "prediction": "ANOMALY" if raw_score >= self.threshold else "NORMAL",
                    "map_max": float(raw_map.max()),
                    "inference_time": inference_time,
                    "model_load_time": model_load_time,
                    "total_time": total_time,
                    "cache_hit": cache_hit,
                    "resize": self.resize,
                    "input_size": MODEL_INPUT_SIZE,
                    "device": str(device),
                    "model_path": str(model_path),
                    "original": display_image,
                    "gt_mask": gt_mask,
                    "heatmap": heatmap,
                    "overlay": overlay,
                }
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ImagePanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("ImagePanel")
        layout = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setObjectName("PanelTitle")
        self.image = QLabel("No image")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(250, 250)
        self.image.setObjectName("ImageLabel")
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)
        self._pixmap = None

    def set_pil_image(self, image):
        if image is None:
            self.image.setText("No image")
            self.image.setPixmap(QPixmap())
            self._pixmap = None
            return
        self._pixmap = pil_to_pixmap(image)
        self._update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(scaled)


class PatchCoreUi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore Defect Detection")
        self.resize(1440, 980)
        self.image_path = ""
        self.mask_path = ""
        self.current_browse_dir = PATCHCORE_ROOT / "data"
        self.worker = None
        self.metric_labels = {}
        self._build_ui()
        self._reset_results()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        layout.addLayout(header)
        title_block = QVBoxLayout()
        title = QLabel("PatchCore Defect Detection")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "Run a category-specific MVTec AD or VisA model and inspect the prediction map."
        )
        subtitle.setObjectName("AppSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)

        controls_frame = QFrame()
        controls_frame.setObjectName("ControlCard")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(18, 14, 18, 14)
        controls_layout.setSpacing(10)
        controls = QGridLayout()
        controls.setHorizontalSpacing(10)
        controls.setVerticalSpacing(8)
        controls_layout.addLayout(controls)
        layout.addWidget(controls_frame)

        dataset_label = QLabel("Dataset")
        dataset_label.setObjectName("FieldLabel")
        self.dataset = QComboBox()
        self.dataset.addItems(list(DATASET_CONFIG.keys()))
        self.dataset.currentTextChanged.connect(self.on_dataset_changed)
        controls.addWidget(dataset_label, 0, 0)
        controls.addWidget(self.dataset, 0, 1)

        category_label = QLabel("Category")
        category_label.setObjectName("FieldLabel")
        self.category = QComboBox()
        self.category.currentTextChanged.connect(self.on_category_changed)
        controls.addWidget(category_label, 0, 2)
        controls.addWidget(self.category, 0, 3)

        resize_label = QLabel("Resize")
        resize_label.setObjectName("FieldLabel")
        self.resize_spin = QSpinBox()
        self.resize_spin.setRange(MODEL_INPUT_SIZE, 2048)
        self.resize_spin.setSingleStep(32)
        self.resize_spin.setSuffix(" px")
        self.resize_spin.setToolTip(
            "Preprocessing resize. The trained PatchCore model still receives a 224 x 224 center crop."
        )
        self.resize_spin.valueChanged.connect(self.on_resize_changed)
        controls.addWidget(resize_label, 0, 4)
        controls.addWidget(self.resize_spin, 0, 5)

        threshold_label = QLabel("Decision threshold")
        threshold_label.setObjectName("FieldLabel")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(1.5)
        self.threshold_spin.setToolTip(
            "Raw PatchCore score threshold. The saved models do not contain a calibrated threshold."
        )
        controls.addWidget(threshold_label, 0, 6)
        controls.addWidget(self.threshold_spin, 0, 7)

        self.image_input = QLineEdit()
        self.image_input.setReadOnly(True)
        image_btn = QPushButton("Load Image")
        image_btn.setObjectName("SecondaryButton")
        image_btn.clicked.connect(self.load_image)
        self.load_image_btn = image_btn
        image_label = QLabel("Image")
        image_label.setObjectName("FieldLabel")
        controls.addWidget(image_label, 1, 0)
        controls.addWidget(self.image_input, 1, 1, 1, 6)
        controls.addWidget(image_btn, 1, 7)

        self.mask_input = QLineEdit()
        self.mask_input.setReadOnly(True)
        mask_btn = QPushButton("Load GT Mask")
        mask_btn.setObjectName("SecondaryButton")
        mask_btn.clicked.connect(self.load_mask)
        clear_mask_btn = QPushButton("Clear Mask")
        clear_mask_btn.setObjectName("GhostButton")
        clear_mask_btn.clicked.connect(self.clear_mask)
        self.load_mask_btn = mask_btn
        self.clear_mask_btn = clear_mask_btn
        mask_label = QLabel("Ground Truth Mask")
        mask_label.setObjectName("FieldLabel")
        controls.addWidget(mask_label, 2, 0)
        controls.addWidget(self.mask_input, 2, 1, 1, 5)
        controls.addWidget(mask_btn, 2, 6)
        controls.addWidget(clear_mask_btn, 2, 7)
        controls.setColumnStretch(1, 1)
        controls.setColumnStretch(3, 1)

        run_row = QHBoxLayout()
        run_row.setSpacing(10)
        controls_layout.addLayout(run_row)
        self.run_btn = QPushButton("Run Detection")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_detection)
        clear_result_btn = QPushButton("Clear Result")
        clear_result_btn.setObjectName("GhostButton")
        clear_result_btn.clicked.connect(self.clear_result)
        self.clear_result_btn = clear_result_btn
        self.status = QLabel("Ready")
        self.status.setObjectName("StatusLabel")
        run_row.addWidget(self.run_btn)
        run_row.addWidget(clear_result_btn)
        run_row.addWidget(self.status, 1)

        result_card = QFrame()
        result_card.setObjectName("ResultCard")
        result_layout = QGridLayout(result_card)
        result_layout.setContentsMargins(14, 10, 14, 10)
        result_layout.setHorizontalSpacing(22)
        result_layout.setVerticalSpacing(5)
        layout.addWidget(result_card)
        metric_specs = [
            ("Prediction", "prediction"),
            ("Defect score", "score"),
            ("Threshold", "threshold"),
            ("Prediction map max", "map_max"),
            ("Inference time", "inference_time"),
            ("Total time", "total_time"),
            ("Resize", "resize"),
            ("Model input", "input_size"),
            ("Device", "device"),
            ("Model", "model"),
        ]
        for index, (caption, key) in enumerate(metric_specs):
            row, column = divmod(index, 5)
            block = QVBoxLayout()
            caption_label = QLabel(caption)
            caption_label.setObjectName("MetricCaption")
            value_label = QLabel("—")
            value_label.setObjectName("MetricValue")
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            block.addWidget(caption_label)
            block.addWidget(value_label)
            result_layout.addLayout(block, row, column)
            self.metric_labels[key] = value_label

        panel_grid = QGridLayout()
        panel_grid.setHorizontalSpacing(14)
        panel_grid.setVerticalSpacing(14)
        layout.addLayout(panel_grid, 1)
        self.original_panel = ImagePanel("Original")
        self.mask_panel = ImagePanel("Ground Truth Mask")
        self.heatmap_panel = ImagePanel("Predicted Heatmap")
        self.overlay_panel = ImagePanel("Overlay")
        panel_grid.addWidget(self.original_panel, 0, 0)
        panel_grid.addWidget(self.mask_panel, 0, 1)
        panel_grid.addWidget(self.heatmap_panel, 1, 0)
        panel_grid.addWidget(self.overlay_panel, 1, 1)

        self.setStyleSheet(
            """
            QMainWindow, #AppRoot { background: #f5f5f7; color: #1d1d1f; }
            QLabel { font-size: 13px; color: #1d1d1f; }
            #AppTitle { font-size: 26px; font-weight: 700; color: #1d1d1f; }
            #AppSubtitle { font-size: 13px; color: #6e6e73; padding-top: 2px; }
            #ControlCard, #ResultCard {
                background: rgba(255, 255, 255, 235);
                border: 1px solid #e5e5ea; border-radius: 14px;
            }
            #FieldLabel { color: #6e6e73; font-weight: 600; padding-right: 6px; }
            QPushButton {
                min-height: 30px; padding: 7px 14px; border-radius: 8px;
                border: 1px solid #d1d1d6; background: #ffffff;
                color: #1d1d1f; font-weight: 600;
            }
            QPushButton:hover { background: #f2f2f7; }
            QPushButton:pressed { background: #e5e5ea; }
            #PrimaryButton { background: #007aff; border: 1px solid #007aff; color: white; padding-left: 18px; padding-right: 18px; }
            #PrimaryButton:hover { background: #0a84ff; }
            #PrimaryButton:pressed { background: #0066d6; }
            #PrimaryButton:disabled { background: #a8cfff; border-color: #a8cfff; }
            #GhostButton { background: transparent; color: #6e6e73; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 30px; padding: 5px 9px; background: #ffffff;
                border: 1px solid #d1d1d6; border-radius: 8px; color: #1d1d1f;
            }
            #ImagePanel { background: #ffffff; border: 1px solid #e5e5ea; border-radius: 14px; }
            #PanelTitle { font-weight: bold; padding: 8px; color: #1d1d1f; }
            #ImageLabel { color: #8e8e93; background: #fbfbfd; border-radius: 10px; }
            #StatusLabel { padding: 8px 12px; color: #3a3a3c; background: #f2f2f7; border-radius: 8px; }
            #MetricCaption { color: #6e6e73; font-size: 12px; }
            #MetricValue { color: #1d1d1f; font-size: 15px; font-weight: 700; }
            """
        )

        self.on_dataset_changed(self.dataset.currentText())

    def _reset_results(self):
        self.original_panel.set_pil_image(None)
        self.mask_panel.set_pil_image(None)
        self.heatmap_panel.set_pil_image(None)
        self.overlay_panel.set_pil_image(None)
        for label in self.metric_labels.values():
            label.setText("—")
        self.metric_labels["prediction"].setStyleSheet("")

    def _set_busy(self, busy):
        for widget in [
            self.dataset,
            self.category,
            self.resize_spin,
            self.threshold_spin,
            self.load_image_btn,
            self.load_mask_btn,
            self.clear_mask_btn,
            self.clear_result_btn,
        ]:
            widget.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)

    def _set_category_items(self, dataset_name):
        current = self.category.currentText()
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItems(dataset_config(dataset_name)["classes"])
        if current in dataset_config(dataset_name)["classes"]:
            self.category.setCurrentText(current)
        self.category.blockSignals(False)

    def on_dataset_changed(self, dataset_name):
        if dataset_name not in DATASET_CONFIG:
            return
        self._set_category_items(dataset_name)
        self.resize_spin.setValue(dataset_config(dataset_name)["default_resize"])
        self.current_browse_dir = dataset_config(dataset_name)["default_folder"]
        self.clear_result()
        self.status.setText(
            f"{dataset_name} selected; choose an image and matching category model"
        )

    def on_category_changed(self, category):
        if category and self.image_path:
            self.status.setText(f"Category changed to {category}; ready to run detection")

    def on_resize_changed(self, resize):
        if not self.image_path:
            return
        try:
            image, _ = preprocess_image(self.image_path, resize)
            self.original_panel.set_pil_image(image)
            if self.mask_path:
                self.mask_panel.set_pil_image(preprocess_mask(self.mask_path, resize))
            self.status.setText(
                f"Preview updated: resize {resize}px → model input {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}"
            )
        except Exception as exc:
            self.status.setText(f"Preview update failed: {exc}")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            str(self.current_browse_dir),
            "Images (*.png *.jpg *.jpeg *.bmp *.JPG *.PNG)",
        )
        if not path:
            return

        self.current_browse_dir = Path(path).parent
        detected_dataset = infer_dataset_from_path(path)
        if detected_dataset and detected_dataset != self.dataset.currentText():
            self.dataset.setCurrentText(detected_dataset)
        detected_category = infer_category_from_path(path, self.dataset.currentText())
        if detected_category:
            self.category.setCurrentText(detected_category)

        self.image_path = path
        self.image_input.setText(path)
        try:
            image, _ = preprocess_image(path, self.resize_spin.value())
            self.original_panel.set_pil_image(image)
            mask_path = find_ground_truth_mask(self.dataset.currentText(), path)
            if mask_path:
                self.mask_path = mask_path
                self.mask_input.setText(mask_path)
                self.mask_panel.set_pil_image(
                    preprocess_mask(mask_path, self.resize_spin.value())
                )
                self.status.setText("Image loaded; matching ground-truth mask found")
            else:
                self.clear_mask()
                self.status.setText("Image loaded; no ground-truth mask found")
            self.clear_prediction_panels()
        except Exception as exc:
            self.image_path = ""
            self.image_input.clear()
            QMessageBox.critical(self, "Image load failed", str(exc))

    def load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ground truth mask",
            str(self.current_browse_dir),
            "Images (*.png *.jpg *.jpeg *.bmp *.JPG *.PNG)",
        )
        if not path:
            return
        self.mask_path = path
        self.mask_input.setText(path)
        self.mask_panel.set_pil_image(preprocess_mask(path, self.resize_spin.value()))
        self.status.setText("Ground-truth mask loaded")

    def clear_mask(self):
        self.mask_path = ""
        self.mask_input.clear()
        self.mask_panel.set_pil_image(None)

    def clear_prediction_panels(self):
        self.heatmap_panel.set_pil_image(None)
        self.overlay_panel.set_pil_image(None)
        for key in [
            "prediction",
            "score",
            "threshold",
            "map_max",
            "inference_time",
            "total_time",
            "resize",
            "input_size",
            "device",
            "model",
        ]:
            if key in self.metric_labels:
                self.metric_labels[key].setText("—")
        self.metric_labels["prediction"].setStyleSheet("")

    def clear_result(self):
        self.clear_prediction_panels()
        if hasattr(self, "status"):
            self.status.setText("Ready")

    def run_detection(self):
        if not self.image_path:
            QMessageBox.warning(self, "Missing image", "Please load an image first.")
            return

        dataset_name = self.dataset.currentText()
        category = self.category.currentText()
        model_path = model_path_for(dataset_name, category)
        if not model_path.exists():
            QMessageBox.critical(self, "Model not found", str(model_path))
            return

        self._set_busy(True)
        self.status.setText(
            f"Loading {dataset_name}/{category} model and running inference..."
        )
        self.worker = InferenceWorker(
            dataset_name=dataset_name,
            category=category,
            image_path=self.image_path,
            mask_path=self.mask_path,
            resize=self.resize_spin.value(),
            threshold=self.threshold_spin.value(),
        )
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.failed.connect(self.on_inference_failed)
        self.worker.start()

    def on_inference_finished(self, result):
        self.original_panel.set_pil_image(result["original"])
        self.mask_panel.set_pil_image(result["gt_mask"])
        self.heatmap_panel.set_pil_image(result["heatmap"])
        self.overlay_panel.set_pil_image(result["overlay"])

        self.metric_labels["prediction"].setText(result["prediction"])
        prediction_color = "#c62828" if result["prediction"] == "ANOMALY" else "#188038"
        self.metric_labels["prediction"].setStyleSheet(
            f"color: {prediction_color}; font-size: 15px; font-weight: 700;"
        )
        self.metric_labels["score"].setText(f"{result['score']:.6f}")
        self.metric_labels["threshold"].setText(f"{result['threshold']:.4f}")
        self.metric_labels["map_max"].setText(f"{result['map_max']:.6f}")
        self.metric_labels["inference_time"].setText(
            f"{result['inference_time']:.3f} s"
        )
        self.metric_labels["total_time"].setText(f"{result['total_time']:.3f} s")
        self.metric_labels["resize"].setText(f"{result['resize']} px")
        self.metric_labels["input_size"].setText(
            f"{result['input_size']} x {result['input_size']}"
        )
        self.metric_labels["device"].setText(result["device"])
        self.metric_labels["model"].setText(
            f"{result['dataset']}/{result['category']}"
        )
        self.metric_labels["model"].setToolTip(
            f"{result['model_path']}\nModel load: {result['model_load_time']:.3f}s\n"
            f"Cache hit: {'yes' if result['cache_hit'] else 'no'}"
        )

        cache_text = "cached model" if result["cache_hit"] else "model loaded"
        self.status.setText(
            f"{result['dataset']} / {result['category']} | "
            f"Prediction: {result['prediction']} | {cache_text}"
        )
        self._set_busy(False)
        self.worker.deleteLater()

    def on_inference_failed(self, message):
        self._set_busy(False)
        self.status.setText("Inference failed")
        QMessageBox.critical(self, "Inference failed", message)
        if self.worker is not None:
            self.worker.deleteLater()


def main():
    app = QApplication(sys.argv)
    window = PatchCoreUi()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
