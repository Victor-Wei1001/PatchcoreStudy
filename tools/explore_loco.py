from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\data\mvtec_loco_anomaly_detection")
OUT_DIR = Path(r"D:\patchcore\outputs\loco_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

ANOMALY_TYPES = ["logical_anomalies", "structural_anomalies"]


def list_images_recursive(folder: Path):
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])


def read_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def calculate_mask_area(mask_path: Path) -> int:
    """
    Mask area = number of non-zero pixels.
    This is consistent with our previous MVTec AD / VisA / BTAD statistics.
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    return int((mask_array > 0).sum())


def get_categories():
    """
    MVTec LOCO AD categories are top-level folders.
    Exclude txt/license files.
    """
    return sorted([
        p.name for p in ROOT.iterdir()
        if p.is_dir()
    ])


def find_mask_for_anomaly_image(category: str, anomaly_type: str, image_path: Path):
    """
    Expected LOCO-like structure:

    image:
        category/test/logical_anomalies/000.png
        or
        category/test/logical_anomalies/000/000.png

    mask:
        category/ground_truth/logical_anomalies/000/000.png

    This function handles both possible image layouts.
    """
    test_type_dir = ROOT / category / "test" / anomaly_type
    gt_type_dir = ROOT / category / "ground_truth" / anomaly_type

    if not gt_type_dir.exists():
        return None

    try:
        rel = image_path.relative_to(test_type_dir)
    except ValueError:
        return None

    rel_parts = list(rel.parts)

    # Case A:
    # test/logical_anomalies/000.png
    # sample_id = 000
    if len(rel_parts) == 1:
        sample_id = image_path.stem
        image_filename = image_path.name
        image_stem = image_path.stem

    # Case B:
    # test/logical_anomalies/000/000.png
    # sample_id = 000
    else:
        sample_id = rel_parts[0]
        image_filename = rel_parts[-1]
        image_stem = image_path.stem

    candidate_paths = []

    # Most likely LOCO format
    candidate_paths.append(gt_type_dir / sample_id / image_filename)
    candidate_paths.append(gt_type_dir / sample_id / f"{image_stem}.png")
    candidate_paths.append(gt_type_dir / sample_id / "000.png")

    # Fallbacks
    candidate_paths.append(gt_type_dir / f"{sample_id}.png")
    candidate_paths.append(gt_type_dir / f"{image_stem}.png")

    for ext in IMG_EXTS:
        candidate_paths.append(gt_type_dir / sample_id / f"{image_stem}{ext}")
        candidate_paths.append(gt_type_dir / sample_id / f"{image_stem}_mask{ext}")
        candidate_paths.append(gt_type_dir / sample_id / f"{image_stem}_gt{ext}")
        candidate_paths.append(gt_type_dir / f"{sample_id}{ext}")
        candidate_paths.append(gt_type_dir / f"{sample_id}_mask{ext}")

    for p in candidate_paths:
        if p.exists():
            return p

    # Last fallback:
    # if ground_truth/anomaly_type/sample_id/ contains exactly one image, use it.
    sample_mask_dir = gt_type_dir / sample_id
    if sample_mask_dir.exists() and sample_mask_dir.is_dir():
        mask_candidates = list_images_recursive(sample_mask_dir)
        if len(mask_candidates) == 1:
            return mask_candidates[0]

    return None


def main():
    print("===== MVTec LOCO AD exploration =====")
    print("ROOT:", ROOT)
    print("Exists:", ROOT.exists())

    if not ROOT.exists():
        print("[ERROR] Root path does not exist.")
        return

    categories = get_categories()
    print("\nCategories:")
    for c in categories:
        print("-", c)

    image_rows = []
    defect_rows = []
    missing_mask_rows = []

    category_rows = []

    for category in categories:
        category_dir = ROOT / category

        train_good_dir = category_dir / "train" / "good"
        validation_good_dir = category_dir / "validation" / "good"
        test_good_dir = category_dir / "test" / "good"

        train_good_images = list_images_recursive(train_good_dir)
        validation_good_images = list_images_recursive(validation_good_dir)
        test_good_images = list_images_recursive(test_good_dir)

        # Normal train images
        for image_path in train_good_images:
            image_rows.append({
                "dataset": "MVTec_LOCO_AD",
                "category": category,
                "split": "train",
                "label": "normal",
                "anomaly_type": "none",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
            })

        # Normal validation images
        for image_path in validation_good_images:
            image_rows.append({
                "dataset": "MVTec_LOCO_AD",
                "category": category,
                "split": "validation",
                "label": "normal",
                "anomaly_type": "none",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
            })

        # Normal test images
        for image_path in test_good_images:
            image_rows.append({
                "dataset": "MVTec_LOCO_AD",
                "category": category,
                "split": "test",
                "label": "normal",
                "anomaly_type": "none",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
            })

        anomaly_count_by_type = {}
        mask_count_by_type = {}

        for anomaly_type in ANOMALY_TYPES:
            test_anomaly_dir = category_dir / "test" / anomaly_type
            gt_anomaly_dir = category_dir / "ground_truth" / anomaly_type

            test_anomaly_images = list_images_recursive(test_anomaly_dir)
            gt_mask_images = list_images_recursive(gt_anomaly_dir)

            anomaly_count_by_type[anomaly_type] = len(test_anomaly_images)
            mask_count_by_type[anomaly_type] = len(gt_mask_images)

            for image_path in test_anomaly_images:
                mask_path = find_mask_for_anomaly_image(category, anomaly_type, image_path)
                has_mask = mask_path is not None

                image_rows.append({
                    "dataset": "MVTec_LOCO_AD",
                    "category": category,
                    "split": "test",
                    "label": "anomaly",
                    "anomaly_type": anomaly_type,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path) if has_mask else "",
                    "has_mask": has_mask,
                })

                if not has_mask:
                    missing_mask_rows.append({
                        "category": category,
                        "anomaly_type": anomaly_type,
                        "image_path": str(image_path),
                    })
                    continue

                image_width, image_height = read_image_size(image_path)
                image_area_pixels = image_width * image_height

                defect_area_pixels = calculate_mask_area(mask_path)
                defect_area_ratio = defect_area_pixels / image_area_pixels
                defect_area_percent = defect_area_ratio * 100

                defect_rows.append({
                    "dataset": "MVTec_LOCO_AD",
                    "category": category,
                    "anomaly_type": anomaly_type,
                    "split": "test",
                    "label": "anomaly",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_area_pixels": image_area_pixels,
                    "defect_area_pixels": defect_area_pixels,
                    "defect_area_ratio": defect_area_ratio,
                    "defect_area_percent_of_full_image": defect_area_percent,
                })

        category_rows.append({
            "dataset": "MVTec_LOCO_AD",
            "category": category,

            "train_normal": len(train_good_images),
            "validation_normal": len(validation_good_images),
            "test_normal": len(test_good_images),

            "test_logical_anomaly": anomaly_count_by_type.get("logical_anomalies", 0),
            "test_structural_anomaly": anomaly_count_by_type.get("structural_anomalies", 0),

            "logical_mask_files": mask_count_by_type.get("logical_anomalies", 0),
            "structural_mask_files": mask_count_by_type.get("structural_anomalies", 0),

            "matched_logical_masks": len([
                r for r in defect_rows
                if r["category"] == category and r["anomaly_type"] == "logical_anomalies"
            ]),
            "matched_structural_masks": len([
                r for r in defect_rows
                if r["category"] == category and r["anomaly_type"] == "structural_anomalies"
            ]),

            "total_original_images": (
                len(train_good_images)
                + len(validation_good_images)
                + len(test_good_images)
                + anomaly_count_by_type.get("logical_anomalies", 0)
                + anomaly_count_by_type.get("structural_anomalies", 0)
            ),
        })

        print(f"\nCategory: {category}")
        print("train/good:", len(train_good_images))
        print("validation/good:", len(validation_good_images))
        print("test/good:", len(test_good_images))
        print("test/logical_anomalies:", anomaly_count_by_type.get("logical_anomalies", 0))
        print("test/structural_anomalies:", anomaly_count_by_type.get("structural_anomalies", 0))
        print("gt/logical_anomalies masks:", mask_count_by_type.get("logical_anomalies", 0))
        print("gt/structural_anomalies masks:", mask_count_by_type.get("structural_anomalies", 0))

    image_df = pd.DataFrame(image_rows)
    defect_df = pd.DataFrame(defect_rows)
    category_df = pd.DataFrame(category_rows).sort_values("category")

    image_index_csv = OUT_DIR / "loco_image_files_index.csv"
    category_csv = OUT_DIR / "loco_category_stats.csv"

    image_df.to_csv(image_index_csv, index=False, encoding="utf-8-sig")
    category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")

    print("\n===== LOCO category statistics =====")
    print(category_df)

    print("\nSaved:")
    print(image_index_csv)
    print(category_csv)

    # Save missing mask log if needed
    if missing_mask_rows:
        missing_df = pd.DataFrame(missing_mask_rows)
        missing_csv = OUT_DIR / "loco_missing_masks.csv"
        missing_df.to_csv(missing_csv, index=False, encoding="utf-8-sig")

        print("\n[WARNING] Some anomaly images did not match masks.")
        print("Missing masks:", len(missing_df))
        print("Saved:", missing_csv)

    if defect_df.empty:
        print("\n[WARNING] No defect masks matched. Please check directory structure.")
        return

    # Per-image defect size
    per_image_csv = OUT_DIR / "loco_defect_size_per_image.csv"
    defect_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    # Category-wise defect size
    defect_category_df = (
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
        .sort_values("avg_defect_area_percent_of_full_image", ascending=False)
    )

    defect_category_csv = OUT_DIR / "loco_defect_size_categorywise.csv"
    defect_category_df.to_csv(defect_category_csv, index=False, encoding="utf-8-sig")

    # Overall anomaly-type-wise defect size
    defect_anomalytype_df = (
        defect_df
        .groupby("anomaly_type")
        .agg(
            defect_mask_count=("mask_path", "count"),
            avg_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "mean"),
            median_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "median"),
            min_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "min"),
            max_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "max"),
            std_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "std"),
        )
        .reset_index()
        .sort_values("avg_defect_area_percent_of_full_image", ascending=False)
    )

    defect_anomalytype_csv = OUT_DIR / "loco_defect_size_anomalytypewise.csv"
    defect_anomalytype_df.to_csv(defect_anomalytype_csv, index=False, encoding="utf-8-sig")

    # Category + anomaly-type-wise defect size
    defect_category_anomalytype_df = (
        defect_df
        .groupby(["category", "anomaly_type"])
        .agg(
            defect_mask_count=("mask_path", "count"),
            avg_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "mean"),
            median_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "median"),
            min_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "min"),
            max_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "max"),
            std_defect_area_percent_of_full_image=("defect_area_percent_of_full_image", "std"),
        )
        .reset_index()
        .sort_values(["category", "avg_defect_area_percent_of_full_image"], ascending=[True, False])
    )

    defect_category_anomalytype_csv = OUT_DIR / "loco_defect_size_category_anomalytypewise.csv"
    defect_category_anomalytype_df.to_csv(defect_category_anomalytype_csv, index=False, encoding="utf-8-sig")

    print("\n===== LOCO defect size category-wise =====")
    print(defect_category_df)

    print("\n===== LOCO defect size anomaly-type-wise =====")
    print(defect_anomalytype_df)

    # Visualization 1: image count stacked bar
    plot_df = category_df.set_index("category")
    stacked_cols = [
        "train_normal",
        "validation_normal",
        "test_normal",
        "test_logical_anomaly",
        "test_structural_anomaly",
    ]

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(plot_df))

    for col in stacked_cols:
        if col in plot_df.columns and plot_df[col].sum() > 0:
            plt.bar(plot_df.index, plot_df[col], bottom=bottom, label=col)
            bottom += plot_df[col].values

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("MVTec LOCO AD category")
    plt.ylabel("Image count")
    plt.title("MVTec LOCO AD Image Count by Category and Split")
    plt.legend()
    plt.tight_layout()

    stacked_bar_path = OUT_DIR / "loco_image_count_stacked_bar.png"
    plt.savefig(stacked_bar_path, dpi=150)
    plt.close()

    # Visualization 2: category-wise average defect area
    plt.figure(figsize=(12, 6))
    plt.bar(
        defect_category_df["category"],
        defect_category_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("MVTec LOCO AD category")
    plt.ylabel("Average defect area / full image area (%)")
    plt.title("MVTec LOCO AD Category-wise Average Defect Area Percentage")
    plt.tight_layout()

    defect_category_bar_path = OUT_DIR / "loco_defect_size_categorywise_bar.png"
    plt.savefig(defect_category_bar_path, dpi=150)
    plt.close()

    # Visualization 3: anomaly-type-wise average defect area
    plt.figure(figsize=(8, 6))
    plt.bar(
        defect_anomalytype_df["anomaly_type"],
        defect_anomalytype_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Anomaly type")
    plt.ylabel("Average defect area / full image area (%)")
    plt.title("MVTec LOCO AD Average Defect Area by Anomaly Type")
    plt.tight_layout()

    defect_anomalytype_bar_path = OUT_DIR / "loco_defect_size_anomalytypewise_bar.png"
    plt.savefig(defect_anomalytype_bar_path, dpi=150)
    plt.close()

    # Visualization 4: boxplot by category
    categories_order = sorted(defect_df["category"].unique())
    box_data = [
        defect_df.loc[
            defect_df["category"] == cat,
            "defect_area_percent_of_full_image"
        ].values
        for cat in categories_order
    ]

    plt.figure(figsize=(12, 6))
    plt.boxplot(box_data, labels=categories_order, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("MVTec LOCO AD category")
    plt.ylabel("Defect area / full image area (%)")
    plt.title("MVTec LOCO AD Defect Area Percentage Distribution by Category")
    plt.tight_layout()

    boxplot_path = OUT_DIR / "loco_defect_size_boxplot.png"
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # Excel report
    excel_path = OUT_DIR / "loco_statistics_report.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            image_df.to_excel(writer, sheet_name="image_index", index=False)
            category_df.to_excel(writer, sheet_name="category_stats", index=False)
            defect_df.to_excel(writer, sheet_name="defect_per_image", index=False)
            defect_category_df.to_excel(writer, sheet_name="defect_categorywise", index=False)
            defect_anomalytype_df.to_excel(writer, sheet_name="defect_anomalytype", index=False)
            defect_category_anomalytype_df.to_excel(writer, sheet_name="category_anomalytype", index=False)

        print("\nExcel report saved:")
        print(excel_path)

    except Exception as e:
        print("\n[WARNING] Excel report was not saved.")
        print("Reason:", e)

    print("\nSaved files:")
    print(per_image_csv)
    print(defect_category_csv)
    print(defect_anomalytype_csv)
    print(defect_category_anomalytype_csv)

    print("\nSaved visualizations:")
    print(stacked_bar_path)
    print(defect_category_bar_path)
    print(defect_anomalytype_bar_path)
    print(boxplot_path)

    print("\n===== Done: MVTec LOCO AD exploration finished =====")


if __name__ == "__main__":
    main()