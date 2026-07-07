from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\data\mvtec")
OUT_DIR = Path(r"D:\patchcore\outputs\dataset_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_corresponding_image(category_dir: Path, defect_type: str, mask_path: Path):
    """
    MVTec AD structure:

    mask:
        category/ground_truth/defect_type/000_mask.png

    image:
        category/test/defect_type/000.png
    """
    test_dir = category_dir / "test" / defect_type

    # 000_mask.png -> 000
    image_stem = mask_path.stem.replace("_mask", "")

    for ext in IMG_EXTS:
        candidate = test_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def calculate_mask_area(mask_path: Path) -> int:
    """
    Defect area = number of non-zero pixels in the ground-truth mask.
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)

    defect_area_pixels = int((mask_array > 0).sum())
    return defect_area_pixels


def get_image_area(image_path: Path) -> int:
    """
    Image area = width * height of the corresponding defective image.
    """
    image = Image.open(image_path)
    width, height = image.size

    image_area_pixels = width * height
    return image_area_pixels, width, height


def main():
    rows = []

    categories = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    for category_dir in categories:
        category = category_dir.name
        gt_dir = category_dir / "ground_truth"

        if not gt_dir.exists():
            continue

        defect_type_dirs = sorted([p for p in gt_dir.iterdir() if p.is_dir()])

        for defect_type_dir in defect_type_dirs:
            defect_type = defect_type_dir.name

            mask_paths = sorted(defect_type_dir.glob("*.png"))

            for mask_path in mask_paths:
                image_path = find_corresponding_image(category_dir, defect_type, mask_path)

                if image_path is None:
                    print(f"[WARNING] No corresponding image found for mask: {mask_path}")
                    continue

                defect_area_pixels = calculate_mask_area(mask_path)
                image_area_pixels, image_width, image_height = get_image_area(image_path)

                defect_area_ratio = defect_area_pixels / image_area_pixels
                defect_area_percent = defect_area_ratio * 100

                rows.append({
                    "category": category,
                    "defect_type": defect_type,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    "defect_area_pixels": defect_area_pixels,
                    "image_area_pixels": image_area_pixels,
                    "defect_area_ratio": defect_area_ratio,
                    "defect_area_percent_of_full_image": defect_area_percent,
                })

    per_image_df = pd.DataFrame(rows)

    if per_image_df.empty:
        print("No valid mask-image pairs found. Please check dataset path.")
        return

    # 1. Per-image report
    per_image_csv = OUT_DIR / "defect_size_per_image.csv"
    per_image_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    # 2. Category-wise report
    category_df = (
        per_image_df
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

    category_csv = OUT_DIR / "defect_size_categorywise.csv"
    category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")

    # 3. Sub-category-wise report
    subcategory_df = (
        per_image_df
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
        .sort_values(
            ["category", "avg_defect_area_percent_of_full_image"],
            ascending=[True, False]
        )
    )

    subcategory_csv = OUT_DIR / "defect_size_subcategorywise.csv"
    subcategory_df.to_csv(subcategory_csv, index=False, encoding="utf-8-sig")

    # 4. Bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(
        category_df["category"],
        category_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("MVTec AD category")
    plt.ylabel("Average defect area / full image area (%)")
    plt.title("Category-wise Average Defect Area Percentage")
    plt.tight_layout()

    bar_path = OUT_DIR / "defect_size_categorywise_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()

    print("\n===== Done =====")
    print("Per-image report:")
    print(per_image_csv)

    print("\nCategory-wise report:")
    print(category_csv)

    print("\nSub-category-wise report:")
    print(subcategory_csv)

    print("\nBar chart:")
    print(bar_path)

    print("\n===== Category-wise summary =====")
    print(category_df)


if __name__ == "__main__":
    main()

 