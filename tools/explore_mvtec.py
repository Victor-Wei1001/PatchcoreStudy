from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def get_categories(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def find_mask(root: Path, category: str, defect_type: str, image_path: Path):
    if defect_type == "good":
        return None

    gt_dir = root / category / "ground_truth" / defect_type
    if not gt_dir.exists():
        return None

    candidates = [
        gt_dir / f"{image_path.stem}_mask.png",
        gt_dir / f"{image_path.stem}.png",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = list(gt_dir.glob(f"{image_path.stem}*"))
    return matches[0] if matches else None


def save_stats(root: Path, out_csv: Path):
    rows = []

    for category in get_categories(root):
        train_good = list_images(root / category / "train" / "good")
        test_root = root / category / "test"
        gt_root = root / category / "ground_truth"

        test_good = 0
        test_defect = 0
        defect_type_counts = {}

        for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
            imgs = list_images(defect_dir)
            defect_type_counts[defect_dir.name] = len(imgs)

            if defect_dir.name == "good":
                test_good += len(imgs)
            else:
                test_defect += len(imgs)

        masks = list_images(gt_root)

        rows.append({
            "category": category,
            "train_good": len(train_good),
            "test_good": test_good,
            "test_defect": test_defect,
            "test_total": test_good + test_defect,
            "mask_total": len(masks),
            "total_images_without_masks": len(train_good) + test_good + test_defect,
            "defect_type_counts": defect_type_counts,
        })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n===== MVTec AD Category Statistics =====")
    print(df)
    print("\nTotal categories:", len(df))
    print("Total images without masks:", df["total_images_without_masks"].sum())
    print("Total masks:", df["mask_total"].sum())
    print("Saved CSV:", out_csv)


def read_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def mask_to_bbox(mask_path: Path):
    if mask_path is None or not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    binary = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def make_overlay(image_path: Path, mask_path: Path):
    img = read_rgb(image_path)

    if mask_path is None:
        return img

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return img

    overlay = img.copy()
    mask_bool = mask > 0

    red_layer = np.zeros_like(img)
    red_layer[:, :, 0] = 255

    alpha = 0.35
    overlay[mask_bool] = (
        alpha * red_layer[mask_bool] + (1 - alpha) * overlay[mask_bool]
    ).astype(np.uint8)

    bbox = mask_to_bbox(mask_path)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return overlay


def collect_defect_samples(root: Path, category: str):
    samples = []
    test_root = root / category / "test"

    for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        if defect_dir.name == "good":
            continue

        for img_path in list_images(defect_dir):
            mask_path = find_mask(root, category, defect_dir.name, img_path)
            samples.append({
                "category": category,
                "defect_type": defect_dir.name,
                "image_path": img_path,
                "mask_path": mask_path,
            })

    return samples


def make_two_category_panel(root: Path, cat_a: str, cat_b: str, n: int, out_path: Path):
    random.seed(0)

    samples_a = collect_defect_samples(root, cat_a)
    samples_b = collect_defect_samples(root, cat_b)

    random.shuffle(samples_a)
    random.shuffle(samples_b)

    n = min(n, len(samples_a), len(samples_b))

    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        for j, sample in enumerate([samples_a[i], samples_b[i]]):
            overlay = make_overlay(sample["image_path"], sample["mask_path"])
            axes[i, j].imshow(overlay)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{sample['category']} / {sample['defect_type']}")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved two-category panel:", out_path)


def make_original_mask_overlay_panel(root: Path, category: str, n: int, out_path: Path):
    random.seed(1)

    samples = collect_defect_samples(root, category)
    random.shuffle(samples)
    samples = samples[:n]

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = np.array([axes])

    for i, sample in enumerate(samples):
        img = read_rgb(sample["image_path"])

        if sample["mask_path"] is not None:
            mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        overlay = make_overlay(sample["image_path"], sample["mask_path"])

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {sample['category']} / {sample['defect_type']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay + Bounding Box")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("Saved original/mask/overlay panel:", out_path)


if __name__ == "__main__":
    root = Path(r"D:\patchcore\data\mvtec")
    out_dir = Path(r"D:\patchcore\outputs\dataset_exploration")

    save_stats(
        root=root,
        out_csv=out_dir / "mvtec_ad_category_stats.csv"
    )

    make_two_category_panel(
        root=root,
        cat_a="bottle",
        cat_b="cable",
        n=5,
        out_path=out_dir / "bottle_vs_cable_5x2_bbox_overlay.png"
    )

    make_original_mask_overlay_panel(
        root=root,
        category="bottle",
        n=5,
        out_path=out_dir / "bottle_original_mask_overlay_bbox.png"
    )