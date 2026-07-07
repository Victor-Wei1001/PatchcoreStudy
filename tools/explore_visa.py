from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\data\VisA")
SPLIT_CSV = ROOT / "split_csv" / "1cls.csv"

OUT_DIR = Path(r"D:\patchcore\outputs\visa_exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def calculate_mask_area(mask_path: Path) -> int:
    """
    VisA masks are pixel-level annotations.
    Non-zero pixels are treated as defect pixels.
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    defect_area_pixels = int((mask_array > 0).sum())
    return defect_area_pixels


def main():
    print("===== Loading VisA split CSV =====")
    print(SPLIT_CSV)

    df = pd.read_csv(SPLIT_CSV)

    print("\nCSV shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Make full paths
    df["image_path"] = df["image"].apply(lambda x: ROOT / x)
    df["mask_path"] = df["mask"].apply(
        lambda x: ROOT / x if isinstance(x, str) and x.strip() != "" else None
    )

    # Check whether files exist
    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())
    df["mask_exists"] = df["mask_path"].apply(lambda p: p.exists() if p is not None else False)

    missing_images = df.loc[~df["image_exists"]]
    if len(missing_images) > 0:
        print("\n[WARNING] Missing images:")
        print(missing_images[["object", "split", "label", "image"]].head())

    # -----------------------------
    # 1. Category-wise image statistics
    # -----------------------------
    category_rows = []

    for obj, g in df.groupby("object"):
        train_normal = len(g[(g["split"] == "train") & (g["label"] == "normal")])
        train_anomaly = len(g[(g["split"] == "train") & (g["label"] == "anomaly")])

        test_normal = len(g[(g["split"] == "test") & (g["label"] == "normal")])
        test_anomaly = len(g[(g["split"] == "test") & (g["label"] == "anomaly")])

        mask_count = int(g["mask_exists"].sum())
        total_images = len(g)

        category_rows.append({
            "dataset": "VisA",
            "category": obj,
            "train_normal": train_normal,
            "train_anomaly": train_anomaly,
            "test_normal": test_normal,
            "test_anomaly": test_anomaly,
            "mask_count": mask_count,
            "total_images": total_images,
        })

    category_df = pd.DataFrame(category_rows).sort_values("category")

    category_csv = OUT_DIR / "visa_category_stats.csv"
    category_df.to_csv(category_csv, index=False, encoding="utf-8-sig")

    print("\n===== VisA category-wise statistics =====")
    print(category_df)
    print("\nSaved:", category_csv)

    # -----------------------------
    # 2. Defect size per image
    # -----------------------------
    defect_rows = []

    anomaly_df = df[(df["label"] == "anomaly") & (df["mask_exists"])].copy()

    for _, row in anomaly_df.iterrows():
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        if not image_path.exists():
            print(f"[WARNING] Missing image: {image_path}")
            continue

        if mask_path is None or not mask_path.exists():
            print(f"[WARNING] Missing mask: {mask_path}")
            continue

        image_width, image_height = read_image_size(image_path)
        image_area_pixels = image_width * image_height

        defect_area_pixels = calculate_mask_area(mask_path)
        defect_area_ratio = defect_area_pixels / image_area_pixels
        defect_area_percent = defect_area_ratio * 100

        defect_rows.append({
            "dataset": "VisA",
            "category": row["object"],
            "defect_type": "anomaly",
            "split": row["split"],
            "label": row["label"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "image_width": image_width,
            "image_height": image_height,
            "image_area_pixels": image_area_pixels,
            "defect_area_pixels": defect_area_pixels,
            "defect_area_ratio": defect_area_ratio,
            "defect_area_percent_of_full_image": defect_area_percent,
        })

    defect_df = pd.DataFrame(defect_rows)

    if defect_df.empty:
        print("\n[WARNING] No anomaly masks found. Defect size report was not generated.")
        return

    per_image_csv = OUT_DIR / "visa_defect_size_per_image.csv"
    defect_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    # -----------------------------
    # 3. Category-wise defect size
    # -----------------------------
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

    defect_category_csv = OUT_DIR / "visa_defect_size_categorywise.csv"
    defect_category_df.to_csv(defect_category_csv, index=False, encoding="utf-8-sig")

    # -----------------------------
    # 4. Sub-category-wise defect size
    # VisA split only gives label=anomaly 
    # -----------------------------
 

    print("\n===== VisA defect size category-wise report =====")
    print(defect_category_df)
    print("\nSaved:")
    print(per_image_csv)
    print(defect_category_csv)
     
    # -----------------------------
    # 5. Visualization: stacked bar chart
    # -----------------------------
    plot_df = category_df.set_index("category")

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(plot_df))

    for col in ["train_normal", "test_normal", "test_anomaly"]:
        plt.bar(plot_df.index, plot_df[col], bottom=bottom, label=col)
        bottom += plot_df[col].values

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Image count")
    plt.xlabel("VisA category")
    plt.title("VisA Image Count by Category and Split")
    plt.legend()
    plt.tight_layout()

    stacked_bar_path = OUT_DIR / "visa_image_count_stacked_bar.png"
    plt.savefig(stacked_bar_path, dpi=150)
    plt.close()

    # -----------------------------
    # 6. Visualization: average defect size bar chart
    # -----------------------------
    plt.figure(figsize=(12, 6))
    plt.bar(
        defect_category_df["category"],
        defect_category_df["avg_defect_area_percent_of_full_image"]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average defect area / full image area (%)")
    plt.xlabel("VisA category")
    plt.title("VisA Category-wise Average Defect Area Percentage")
    plt.tight_layout()

    defect_bar_path = OUT_DIR / "visa_defect_size_categorywise_bar.png"
    plt.savefig(defect_bar_path, dpi=150)
    plt.close()

    # -----------------------------
    # 7. Visualization: defect size boxplot
    # -----------------------------
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
    plt.ylabel("Defect area / full image area (%)")
    plt.xlabel("VisA category")
    plt.title("VisA Defect Area Percentage Distribution by Category")
    plt.tight_layout()

    boxplot_path = OUT_DIR / "visa_defect_size_boxplot.png"
    plt.savefig(boxplot_path, dpi=150)
    plt.close()

    # -----------------------------
    # 8. Optional Excel report
    # -----------------------------
    excel_path = OUT_DIR / "visa_statistics_report.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            category_df.to_excel(writer, sheet_name="category_stats", index=False)
            defect_df.to_excel(writer, sheet_name="defect_per_image", index=False)
            defect_category_df.to_excel(writer, sheet_name="defect_categorywise", index=False)
            

        print("\nExcel report saved:")
        print(excel_path)
    except Exception as e:
        print("\n[WARNING] Excel report was not saved.")
        print("Reason:", e)
        print("CSV reports were saved successfully, so this is not a critical issue.")

    print("\nSaved visualizations:")
    print(stacked_bar_path)
    print(defect_bar_path)
    print(boxplot_path)

    print("\n===== Done: VisA exploration finished =====")


if __name__ == "__main__":
    main()