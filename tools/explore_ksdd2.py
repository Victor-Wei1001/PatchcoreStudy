from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\data\KolektorSDD2")
OUT_DIR = Path(r"D:\patchcore\outputs\ksdd2_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CATEGORY_NAME = "Kolektor_surface"


def list_original_images(split_dir: Path):
    """
    KSDD2 stores original image and GT mask in the same folder:

    20000.png
    20000_GT.png

    Original images are those without '_GT' in the stem.
    """
    if not split_dir.exists():
        return []

    return sorted([
        p for p in split_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in IMG_EXTS
            and not p.stem.endswith("_GT")
        )
    ])


def find_gt_mask(image_path: Path):
    """
    image:
        20000.png

    mask:
        20000_GT.png
    """
    mask_path = image_path.with_name(f"{image_path.stem}_GT{image_path.suffix}")

    if mask_path.exists():
        return mask_path

    # fallback: try png mask even if image extension differs
    fallback = image_path.with_name(f"{image_path.stem}_GT.png")
    if fallback.exists():
        return fallback

    return None


def read_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def calculate_mask_area(mask_path: Path) -> int:
    """
    Defect area = number of non-zero pixels in the GT mask.
    For normal images, the GT mask should be fully black, so area = 0.
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    return int((mask_array > 0).sum())


def process_split(split_name: str):
    split_dir = ROOT / split_name
    image_paths = list_original_images(split_dir)

    image_rows = []
    defect_rows = []
    missing_masks = []

    for image_path in image_paths:
        mask_path = find_gt_mask(image_path)

        if mask_path is None:
            missing_masks.append(str(image_path))
            continue

        image_width, image_height = read_image_size(image_path)
        image_area_pixels = image_width * image_height

        defect_area_pixels = calculate_mask_area(mask_path)
        defect_area_ratio = defect_area_pixels / image_area_pixels
        defect_area_percent = defect_area_ratio * 100

        label = "anomaly" if defect_area_pixels > 0 else "normal"

        image_rows.append({
            "dataset": "KolektorSDD2",
            "category": CATEGORY_NAME,
            "split": split_name,
            "label": label,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "has_mask": True,
            "image_width": image_width,
            "image_height": image_height,
            "image_area_pixels": image_area_pixels,
            "defect_area_pixels": defect_area_pixels,
            "defect_area_ratio": defect_area_ratio,
            "defect_area_percent_of_full_image": defect_area_percent,
        })

        # Only anomalous images go into defect-size report.
        if label == "anomaly":
            defect_rows.append({
                "dataset": "KolektorSDD2",
                "category": CATEGORY_NAME,
                "split": split_name,
                "label": label,
                "defect_type": "surface_defect",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "image_width": image_width,
                "image_height": image_height,
                "image_area_pixels": image_area_pixels,
                "defect_area_pixels": defect_area_pixels,
                "defect_area_ratio": defect_area_ratio,
                "defect_area_percent_of_full_image": defect_area_percent,
            })

    return image_rows, defect_rows, missing_masks


def main():
    print("===== KolektorSDD2 exploration =====")
    print("ROOT:", ROOT)
    print("Exists:", ROOT.exists())

    if not ROOT.exists():
        print("[ERROR] Root path does not exist.")
        return

    all_image_rows = []
    all_defect_rows = []
    all_missing_masks = []

    for split_name in ["train", "test"]:
        image_rows, defect_rows, missing_masks = process_split(split_name)

        all_image_rows.extend(image_rows)
        all_defect_rows.extend(defect_rows)
        all_missing_masks.extend(missing_masks)

        print(f"\nSplit: {split_name}")
        print("Original images:", len(image_rows))
        print("Anomaly images:", len(defect_rows))
        print("Normal images:", len(image_rows) - len(defect_rows))
        print("Missing masks:", len(missing_masks))

    image_df = pd.DataFrame(all_image_rows)
    defect_df = pd.DataFrame(all_defect_rows)

    if image_df.empty:
        print("[ERROR] No original images were found.")
        return

    # Save image index
    image_index_csv = OUT_DIR / "ksdd2_image_files_index.csv"
    image_df.to_csv(image_index_csv, index=False, encoding="utf-8-sig")

    # Save missing masks if any
    if all_missing_masks:
        missing_df = pd.DataFrame({"image_path": all_missing_masks})
        missing_csv = OUT_DIR / "ksdd2_missing_masks.csv"
        missing_df.to_csv(missing_csv, index=False, encoding="utf-8-sig")
        print("\n[WARNING] Some images do not have GT masks.")
        print("Saved:", missing_csv)

    # Dataset/category statistics
    category_rows = []

    category_rows.append({
        "dataset": "KolektorSDD2",
        "category": CATEGORY_NAME,

        "train_normal": len(image_df[(image_df["split"] == "train") & (image_df["label"] == "normal")]),
        "train_anomaly": len(image_df[(image_df["split"] == "train") & (image_df["label"] == "anomaly")]),

        "test_normal": len(image_df[(image_df["split"] == "test") & (image_df["label"] == "normal")]),
        "test_anomaly": len(image_df[(image_df["split"] == "test") & (image_df["label"] == "anomaly")]),

        "total_original_images": len(image_df),
        "gt_mask_file_count": int(image_df["has_mask"].sum()),
        "defect_mask_count": len(defect_df),
    })

    category_df = pd.DataFrame(category_rows)

    category_csv = OUT_DIR / "ksdd2_category_stats.csv"
    category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")

    print("\n===== KSDD2 category statistics =====")
    print(category_df)

    if defect_df.empty:
        print("\n[WARNING] No defect masks with non-zero pixels found.")
        return

    # Per-image defect size
    per_image_csv = OUT_DIR / "ksdd2_defect_size_per_image.csv"
    defect_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    # Overall defect-size summary
    defect_summary_df = (
        defect_df
        .groupby("category")
        .agg(
            defect_mask_count=("mask_path", "count"),
            avg_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "mean"),
            median_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "median"),
            min_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "min"),
            max_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "max"),
            std_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "std"),
        )
        .reset_index()
    )

    defect_summary_csv = OUT_DIR / "ksdd2_defect_size_categorywise.csv"
    defect_summary_df.to_csv(defect_summary_csv, index=False, encoding="utf-8-sig")

    # Split-wise defect-size summary
    defect_splitwise_df = (
        defect_df
        .groupby("split")
        .agg(
            defect_mask_count=("mask_path", "count"),
            avg_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "mean"),
            median_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "median"),
            min_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "min"),
            max_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "max"),
            std_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "std"),
        )
        .reset_index()
    )

    defect_splitwise_csv = OUT_DIR / "ksdd2_defect_size_splitwise.csv"
    defect_splitwise_df.to_csv(defect_splitwise_csv, index=False, encoding="utf-8-sig")

    print("\n===== KSDD2 defect size category-wise =====")
    print(defect_summary_df)

    print("\n===== KSDD2 defect size split-wise =====")
    print(defect_splitwise_df)

    # -----------------------------
    # Visualization 1: train/test normal/anomaly stacked bar
    # -----------------------------
    split_stats = (
        image_df
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["normal", "anomaly"]:
        if col not in split_stats.columns:
            split_stats[col] = 0

    plt.figure(figsize=(8, 6))
    bottom = np.zeros(len(split_stats))

    for col in ["normal", "anomaly"]:
        plt.bar(split_stats["split"], split_stats[col], bottom=bottom, label=col)
        bottom += split_stats[col].values

    plt.xlabel("Split")
    plt.ylabel("Image count")
    plt.title("KolektorSDD2 Image Count by Split and Label")
    plt.legend()
    plt.tight_layout()

    stacked_bar_path = OUT_DIR / "ksdd2_image_count_stacked_bar.png"
    plt.savefig(stacked_bar_path, dpi=150)
    plt.close()

    # -----------------------------
    # Visualization 2: average defect size by split
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(
        defect_splitwise_df["split"],
        defect_splitwise_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xlabel("Split")
    plt.ylabel("Average defect area / full image area (%)")
    plt.title("KolektorSDD2 Average Defect Area Percentage by Split")
    plt.tight_layout()

    split_defect_bar_path = OUT_DIR / "ksdd2_defect_size_splitwise_bar.png"
    plt.savefig(split_defect_bar_path, dpi=150)
    plt.close()

    # -----------------------------
    # Visualization 3: defect size boxplot by split
    # -----------------------------
    split_order = sorted(defect_df["split"].unique())
    box_data = [
        defect_df.loc[
            defect_df["split"] == split,
            "defect_area_percent_of_full_image"
        ].values
        for split in split_order
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(box_data, labels=split_order, showfliers=True)
    plt.xlabel("Split")
    plt.ylabel("Defect area / full image area (%)")
    plt.title("KolektorSDD2 Defect Area Percentage Distribution by Split")
    plt.tight_layout()

    boxplot_path = OUT_DIR / "ksdd2_defect_size_boxplot.png"
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # -----------------------------
    # Visualization 4: histogram of defect size
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.hist(defect_df["defect_area_percent_of_full_image"], bins=30)
    plt.xlabel("Defect area / full image area (%)")
    plt.ylabel("Image count")
    plt.title("KolektorSDD2 Defect Area Percentage Histogram")
    plt.tight_layout()

    hist_path = OUT_DIR / "ksdd2_defect_size_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # Excel report
    excel_path = OUT_DIR / "ksdd2_statistics_report.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            image_df.to_excel(writer, sheet_name="image_index", index=False)
            category_df.to_excel(writer, sheet_name="category_stats", index=False)
            defect_df.to_excel(writer, sheet_name="defect_per_image", index=False)
            defect_summary_df.to_excel(writer, sheet_name="defect_categorywise", index=False)
            defect_splitwise_df.to_excel(writer, sheet_name="defect_splitwise", index=False)

        print("\nExcel report saved:")
        print(excel_path)

    except Exception as e:
        print("\n[WARNING] Excel report was not saved.")
        print("Reason:", e)

    print("\nSaved files:")
    print(image_index_csv)
    print(category_csv)
    print(per_image_csv)
    print(defect_summary_csv)
    print(defect_splitwise_csv)

    print("\nSaved visualizations:")
    print(stacked_bar_path)
    print(split_defect_bar_path)
    print(boxplot_path)
    print(hist_path)

    print("\n===== Done: KolektorSDD2 exploration finished =====")


if __name__ == "__main__":
    main()