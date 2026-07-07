from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\data\BTAD\BTech_Dataset_transformed")
OUT_DIR = Path(r"D:\patchcore\outputs\btad_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["01", "02", "03"]
IMG_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]   

def list_images(folder: Path):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def calculate_mask_area(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    return int((mask_array > 0).sum())


def find_mask_for_anomaly_image(category: str, image_path: Path):
    """
    BTAD structure is usually:

    image:
        category/test/ko/0000.bmp

    mask:
        category/ground_truth/ko/0000.png

    But some categories may use different mask extensions or naming styles.
    This function tries multiple possible mask names.
    """
    gt_dir = ROOT / category / "ground_truth" / "ko"
    stem = image_path.stem

    possible_names = []

    for ext in IMG_EXTS:
        possible_names.extend([
            f"{stem}{ext}",
            f"{stem}_mask{ext}",
            f"{stem}_gt{ext}",
        ])

    for name in possible_names:
        mask_path = gt_dir / name
        if mask_path.exists():
            return mask_path

    return None


def main():
    image_rows = []
    defect_rows = []

    for category in CATEGORIES:
        category_dir = ROOT / category

        train_ok_dir = category_dir / "train" / "ok"
        test_ok_dir = category_dir / "test" / "ok"
        test_ko_dir = category_dir / "test" / "ko"
        gt_ko_dir = category_dir / "ground_truth" / "ko"

        train_ok_images = list_images(train_ok_dir)
        test_ok_images = list_images(test_ok_dir)
        test_ko_images = list_images(test_ko_dir)
        mask_images = list_images(gt_ko_dir)

        # Normal training images
        for image_path in train_ok_images:
            image_rows.append({
                "dataset": "BTAD",
                "category": category,
                "split": "train",
                "label": "normal",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
            })

        # Normal test images
        for image_path in test_ok_images:
            image_rows.append({
                "dataset": "BTAD",
                "category": category,
                "split": "test",
                "label": "normal",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
            })

        # Anomaly test images
        for image_path in test_ko_images:
            mask_path = find_mask_for_anomaly_image(category, image_path)
            has_mask = mask_path is not None

            image_rows.append({
                "dataset": "BTAD",
                "category": category,
                "split": "test",
                "label": "anomaly",
                "image_path": str(image_path),
                "mask_path": str(mask_path) if has_mask else "",
                "has_mask": has_mask,
            })

            if has_mask:
                image_width, image_height = read_image_size(image_path)
                image_area_pixels = image_width * image_height

                defect_area_pixels = calculate_mask_area(mask_path)
                defect_area_ratio = defect_area_pixels / image_area_pixels
                defect_area_percent = defect_area_ratio * 100

                defect_rows.append({
                    "dataset": "BTAD",
                    "category": category,
                    "defect_type": "ko",
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

        print(f"\nCategory {category}")
        print(f"train/ok: {len(train_ok_images)}")
        print(f"test/ok: {len(test_ok_images)}")
        print(f"test/ko: {len(test_ko_images)}")
        print(f"ground_truth/ko masks: {len(mask_images)}")

    image_df = pd.DataFrame(image_rows)
    defect_df = pd.DataFrame(defect_rows)

    image_index_csv = OUT_DIR / "btad_image_files_index.csv"
    image_df.to_csv(image_index_csv, index=False, encoding="utf-8-sig")

    # Category-wise image statistics
    category_rows = []

    for category, g in image_df.groupby("category"):
        category_rows.append({
            "dataset": "BTAD",
            "category": category,
            "train_normal": len(g[(g["split"] == "train") & (g["label"] == "normal")]),
            "test_normal": len(g[(g["split"] == "test") & (g["label"] == "normal")]),
            "test_anomaly": len(g[(g["split"] == "test") & (g["label"] == "anomaly")]),
            "mask_count": int(g["has_mask"].sum()),
            "total_original_images": len(g),
        })

    category_df = pd.DataFrame(category_rows).sort_values("category")

    category_csv = OUT_DIR / "btad_category_stats.csv"
    category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")

    if defect_df.empty:
        print("\n[WARNING] No defect masks found or matched.")
        return

    per_image_csv = OUT_DIR / "btad_defect_size_per_image.csv"
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

    defect_category_csv = OUT_DIR / "btad_defect_size_categorywise.csv"
    defect_category_df.to_csv(defect_category_csv, index=False, encoding="utf-8-sig")

    # BTAD has only ko anomaly folder, so defect-type-wise is basically ko only.
    defect_type_df = (
        defect_df
        .groupby(["category", "defect_type"])
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

    defect_type_csv = OUT_DIR / "btad_defect_size_defecttypewise.csv"
    defect_type_df.to_csv(defect_type_csv, index=False, encoding="utf-8-sig")

    print("\n===== BTAD category statistics =====")
    print(category_df)

    print("\n===== BTAD defect size category-wise =====")
    print(defect_category_df)

    # Visualization 1: stacked image count bar
    plot_df = category_df.set_index("category")
    stacked_cols = ["train_normal", "test_normal", "test_anomaly"]

    plt.figure(figsize=(8, 6))
    bottom = np.zeros(len(plot_df))

    for col in stacked_cols:
        plt.bar(plot_df.index, plot_df[col], bottom=bottom, label=col)
        bottom += plot_df[col].values

    plt.xlabel("BTAD category")
    plt.ylabel("Image count")
    plt.title("BTAD Image Count by Category and Split")
    plt.legend()
    plt.tight_layout()

    stacked_bar_path = OUT_DIR / "btad_image_count_stacked_bar.png"
    plt.savefig(stacked_bar_path, dpi=150)
    plt.close()

    # Visualization 2: average defect size bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(
        defect_category_df["category"],
        defect_category_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xlabel("BTAD category")
    plt.ylabel("Average defect area / full image area (%)")
    plt.title("BTAD Category-wise Average Defect Area Percentage")
    plt.tight_layout()

    defect_bar_path = OUT_DIR / "btad_defect_size_categorywise_bar.png"
    plt.savefig(defect_bar_path, dpi=150)
    plt.close()

    # Visualization 3: defect size boxplot
    categories_order = sorted(defect_df["category"].unique())
    box_data = [
        defect_df.loc[
            defect_df["category"] == cat,
            "defect_area_percent_of_full_image"
        ].values
        for cat in categories_order
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(box_data, labels=categories_order, showfliers=True)
    plt.xlabel("BTAD category")
    plt.ylabel("Defect area / full image area (%)")
    plt.title("BTAD Defect Area Percentage Distribution by Category")
    plt.tight_layout()

    boxplot_path = OUT_DIR / "btad_defect_size_boxplot.png"
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # Excel report
    excel_path = OUT_DIR / "btad_statistics_report.xlsx"

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            image_df.to_excel(writer, sheet_name="image_index", index=False)
            category_df.to_excel(writer, sheet_name="category_stats", index=False)
            defect_df.to_excel(writer, sheet_name="defect_per_image", index=False)
            defect_category_df.to_excel(writer, sheet_name="defect_categorywise", index=False)
            defect_type_df.to_excel(writer, sheet_name="defect_typewise", index=False)

        print("\nExcel report saved:")
        print(excel_path)

    except Exception as e:
        print("\n[WARNING] Excel report was not saved.")
        print("Reason:", e)

    print("\nSaved files:")
    print(image_index_csv)
    print(category_csv)
    print(per_image_csv)
    print(defect_category_csv)
    print(defect_type_csv)

    print("\nSaved visualizations:")
    print(stacked_bar_path)
    print(defect_bar_path)
    print(boxplot_path)

    print("\n===== Done: BTAD exploration finished =====")


if __name__ == "__main__":
    main()