from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(r"D:\patchcore\outputs")
OUT_DIR = ROOT / "five_dataset_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {
        "name": "MVTec AD",
        "folder": ROOT / "MVTec AD_exploration",
        "category_stats": "mvtec_ad_category_stats.csv",
        "defect_per_image": "defect_size_per_image.csv",
    },
    {
        "name": "VisA",
        "folder": ROOT / "visa_exploration",
        "category_stats": "visa_category_stats.csv",
        "defect_per_image": "visa_defect_size_per_image.csv",
    },
    {
        "name": "BTAD",
        "folder": ROOT / "btad_exploration",
        "category_stats": "btad_category_stats.csv",
        "defect_per_image": "btad_defect_size_per_image.csv",
    },
    {
        "name": "KolektorSDD2",
        "folder": ROOT / "ksdd2_exploration",
        "category_stats": "ksdd2_category_stats.csv",
        "defect_per_image": "ksdd2_defect_size_per_image.csv",
    },
    {
        "name": "MVTec LOCO AD",
        "folder": ROOT / "loco_exploration",
        "category_stats": "loco_category_stats.csv",
        "defect_per_image": "loco_defect_size_per_image.csv",
    },
]


def safe_sum(df, columns):
    """
    Sum the first existing columns from a list.
    If none of the columns exist, return 0.
    """
    total = 0
    for col in columns:
        if col in df.columns:
            total += df[col].fillna(0).sum()
    return int(total)


def get_total_original_images(category_df, dataset_name):
    """
    Different datasets use different column names.
    This function normalizes total image count.
    """
    if "total_original_images" in category_df.columns:
        return int(category_df["total_original_images"].fillna(0).sum())

    if "total_images" in category_df.columns:
        return int(category_df["total_images"].fillna(0).sum())

    if dataset_name == "MVTec AD":
        train_good = safe_sum(category_df, ["train_good"])
        test_good = safe_sum(category_df, ["test_good"])
        test_defect = safe_sum(category_df, ["test_defect"])
        return train_good + test_good + test_defect

    return 0


def get_mask_count(category_df, defect_df):
    """
    Prefer the number of masks actually used in defect-size calculation.
    This is the most consistent value across datasets.
    """
    if defect_df is not None and len(defect_df) > 0:
        return int(len(defect_df))

    mask_columns = [
        "mask_total",
        "mask_count",
        "defect_mask_count",
        "gt_mask_file_count",
        "matched_logical_masks",
        "matched_structural_masks",
        "logical_mask_files",
        "structural_mask_files",
    ]
    return safe_sum(category_df, mask_columns)


def get_defect_type_count(defect_df):
    """
    Count defect/anomaly sub-types.
    For LOCO, anomaly_type is used.
    For most other datasets, defect_type is used.
    """
    if defect_df is None or len(defect_df) == 0:
        return 0

    if "defect_type" in defect_df.columns:
        return int(defect_df["defect_type"].dropna().nunique())

    if "anomaly_type" in defect_df.columns:
        return int(defect_df["anomaly_type"].dropna().nunique())

    if "label" in defect_df.columns:
        return int(defect_df["label"].dropna().nunique())

    return 0


def get_defect_area_stats(defect_df):
    """
    Calculate overall defect size statistics based on full image area.
    """
    empty_result = {
        "analyzed_defect_masks": 0,
        "avg_defect_area_percent": 0,
        "median_defect_area_percent": 0,
        "min_defect_area_percent": 0,
        "max_defect_area_percent": 0,
        "std_defect_area_percent": 0,
    }

    if defect_df is None or len(defect_df) == 0:
        return empty_result

    col = "defect_area_percent_of_full_image"

    if col not in defect_df.columns:
        return empty_result

    values = pd.to_numeric(defect_df[col], errors="coerce").dropna()

    if len(values) == 0:
        return empty_result

    return {
        "analyzed_defect_masks": int(len(values)),
        "avg_defect_area_percent": float(values.mean()),
        "median_defect_area_percent": float(values.median()),
        "min_defect_area_percent": float(values.min()),
        "max_defect_area_percent": float(values.max()),
        "std_defect_area_percent": float(values.std()),
    }


def build_summary():
    summary_rows = []
    all_defect_rows = []
    all_category_rows = []

    for item in DATASETS:
        dataset_name = item["name"]
        folder = item["folder"]
        category_path = folder / item["category_stats"]
        defect_path = folder / item["defect_per_image"]

        if not category_path.exists():
            raise FileNotFoundError(f"Missing category stats file: {category_path}")

        category_df = pd.read_csv(category_path)
        category_df["dataset_name_for_comparison"] = dataset_name
        all_category_rows.append(category_df)

        if defect_path.exists():
            defect_df = pd.read_csv(defect_path)
            defect_df["dataset_name_for_comparison"] = dataset_name
            all_defect_rows.append(defect_df)
        else:
            defect_df = None

        train_normal = safe_sum(category_df, ["train_good", "train_normal"])
        train_anomaly = safe_sum(category_df, ["train_anomaly"])

        validation_normal = safe_sum(category_df, ["validation_normal"])

        test_normal = safe_sum(category_df, ["test_good", "test_normal"])

        test_anomaly = safe_sum(
            category_df,
            [
                "test_defect",
                "test_anomaly",
                "test_logical_anomaly",
                "test_structural_anomaly",
            ],
        )

        logical_anomaly = safe_sum(category_df, ["test_logical_anomaly"])
        structural_anomaly = safe_sum(category_df, ["test_structural_anomaly"])

        category_count = int(category_df["category"].dropna().nunique()) if "category" in category_df.columns else len(category_df)

        total_original_images = get_total_original_images(category_df, dataset_name)
        mask_count = get_mask_count(category_df, defect_df)
        defect_type_count = get_defect_type_count(defect_df)
        area_stats = get_defect_area_stats(defect_df)

        summary_rows.append({
            "dataset": dataset_name,
            "category_count": category_count,
            "defect_type_count": defect_type_count,
            "train_normal": train_normal,
            "train_anomaly": train_anomaly,
            "validation_normal": validation_normal,
            "test_normal": test_normal,
            "test_anomaly": test_anomaly,
            "test_logical_anomaly": logical_anomaly,
            "test_structural_anomaly": structural_anomaly,
            "total_original_images": total_original_images,
            "analyzed_defect_masks": mask_count,
            "avg_defect_area_percent": area_stats["avg_defect_area_percent"],
            "median_defect_area_percent": area_stats["median_defect_area_percent"],
            "min_defect_area_percent": area_stats["min_defect_area_percent"],
            "max_defect_area_percent": area_stats["max_defect_area_percent"],
            "std_defect_area_percent": area_stats["std_defect_area_percent"],
        })

    summary_df = pd.DataFrame(summary_rows)

    all_defects_df = pd.concat(all_defect_rows, ignore_index=True) if all_defect_rows else pd.DataFrame()
    all_categories_df = pd.concat(all_category_rows, ignore_index=True) if all_category_rows else pd.DataFrame()

    return summary_df, all_defects_df, all_categories_df


def save_table(summary_df, all_defects_df, all_categories_df):
    csv_path = OUT_DIR / "five_dataset_macro_comparison_summary.csv"
    xlsx_path = OUT_DIR / "five_dataset_macro_comparison_report.xlsx"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="macro_summary")
        all_categories_df.to_excel(writer, index=False, sheet_name="all_category_stats")
        all_defects_df.to_excel(writer, index=False, sheet_name="all_defect_per_image")

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved Excel report: {xlsx_path}")


def plot_total_images(summary_df):
    plot_df = summary_df.sort_values("total_original_images", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["dataset"], plot_df["total_original_images"])
    plt.xlabel("Number of original images")
    plt.ylabel("Dataset")
    plt.title("Total Original Images Across Five Datasets")
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_total_original_images.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def plot_image_split_stacked(summary_df):
    plot_df = summary_df.set_index("dataset")

    columns = [
        "train_normal",
        "train_anomaly",
        "validation_normal",
        "test_normal",
        "test_anomaly",
    ]

    existing_columns = [col for col in columns if col in plot_df.columns]
    plot_df = plot_df[existing_columns]

    plt.figure(figsize=(12, 7))
    bottom = None

    for col in existing_columns:
        values = plot_df[col]

        if bottom is None:
            plt.bar(plot_df.index, values, label=col)
            bottom = values.copy()
        else:
            plt.bar(plot_df.index, values, bottom=bottom, label=col)
            bottom = bottom + values

    plt.xlabel("Dataset")
    plt.ylabel("Image count")
    plt.title("Image Split Comparison Across Five Datasets")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_image_split_stacked_bar.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def plot_category_and_defect_type_count(summary_df):
    plot_df = summary_df.set_index("dataset")[["category_count", "defect_type_count"]]

    plt.figure(figsize=(11, 6))
    plot_df.plot(kind="bar", figsize=(11, 6))

    plt.xlabel("Dataset")
    plt.ylabel("Count")
    plt.title("Category Count and Defect Type Count Across Five Datasets")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_category_and_defect_type_count.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def plot_mask_count(summary_df):
    plot_df = summary_df.sort_values("analyzed_defect_masks", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["dataset"], plot_df["analyzed_defect_masks"])
    plt.xlabel("Number of analyzed defect masks")
    plt.ylabel("Dataset")
    plt.title("Analyzed Ground-Truth Masks Across Five Datasets")
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_analyzed_mask_count.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def plot_average_defect_area(summary_df):
    plot_df = summary_df.sort_values("avg_defect_area_percent", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["dataset"], plot_df["avg_defect_area_percent"])
    plt.xlabel("Average defect area percentage of full image (%)")
    plt.ylabel("Dataset")
    plt.title("Average Defect Area Ratio Across Five Datasets")
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_avg_defect_area_percent.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def plot_defect_area_boxplot(all_defects_df):
    if all_defects_df.empty:
        print("No defect per-image data found. Skip boxplot.")
        return

    col = "defect_area_percent_of_full_image"

    if col not in all_defects_df.columns:
        print(f"Column not found: {col}. Skip boxplot.")
        return

    clean_df = all_defects_df[["dataset_name_for_comparison", col]].copy()
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna()

    datasets = clean_df["dataset_name_for_comparison"].unique().tolist()
    data = [
        clean_df.loc[clean_df["dataset_name_for_comparison"] == dataset, col].values
        for dataset in datasets
    ]

    plt.figure(figsize=(12, 7))
    plt.boxplot(data, labels=datasets, showfliers=False)
    plt.xlabel("Dataset")
    plt.ylabel("Defect area percentage of full image (%)")
    plt.title("Defect Area Distribution Across Five Datasets")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    save_path = OUT_DIR / "comparison_defect_area_boxplot.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved chart: {save_path}")


def main():
    summary_df, all_defects_df, all_categories_df = build_summary()

    print("\nFive-dataset macro comparison summary:")
    print(summary_df)

    save_table(summary_df, all_defects_df, all_categories_df)

    plot_total_images(summary_df)
    plot_image_split_stacked(summary_df)
    plot_category_and_defect_type_count(summary_df)
    plot_mask_count(summary_df)
    plot_average_defect_area(summary_df)
    plot_defect_area_boxplot(all_defects_df)

    print("\nDone.")
    print(f"All comparison outputs are saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()