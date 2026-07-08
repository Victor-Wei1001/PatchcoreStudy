from pathlib import Path
import os
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from viz_helpers import make_original_mask_overlay_panel


OUT_DIR = Path(r"D:\patchcore\outputs\sample_visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def replace_path_prefix(value, old_prefix, new_prefix):
    value = str(value)
    old_prefix = str(old_prefix)
    new_prefix = str(new_prefix)

    if value.lower().startswith(old_prefix.lower()):
        return new_prefix + value[len(old_prefix):]

    return value


def fix_mvtec_paths(df):
    old_root = Path(r"D:\patchcore\data\mvtec")
    actual_root = Path(r"D:\patchcore\data\mvtecAD")

    if old_root.exists() or not actual_root.exists():
        return df

    df = df.copy()
    for col in ["image_path", "mask_path"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda value: replace_path_prefix(value, old_root, actual_root)
            )

    return df


def sample_evenly_by_group(df, group_col, n_per_group=1, max_total=5):
    """
    Take samples evenly from each group.
    Useful for datasets with multiple categories.
    """
    selected = []

    if group_col not in df.columns:
        return df.head(max_total)

    for _, g in df.groupby(group_col):
        selected.append(g.head(n_per_group))

    if len(selected) == 0:
        return df.head(max_total)

    result = pd.concat(selected, ignore_index=True)
    return result.head(max_total)


def build_records(df, title_cols):
    records = []

    for _, row in df.iterrows():
        title_parts = []

        for col in title_cols:
            if col in row and pd.notna(row[col]):
                title_parts.append(str(row[col]))

        row_title = " / ".join(title_parts)

        records.append({
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "row_title": row_title,
        })

    return records


def generate_btad_panel():
    csv_path = Path(r"D:\patchcore\outputs\btad_exploration\btad_defect_size_per_image.csv")

    if not csv_path.exists():
        print("[SKIP] BTAD CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    # Take samples from product 01/02/03.
    df_sample = sample_evenly_by_group(df, group_col="category", n_per_group=2, max_total=6)

    records = build_records(df_sample, title_cols=["category", "defect_type"])

    make_original_mask_overlay_panel(
        records=records,
        save_path=OUT_DIR / "btad_sample_defects_panel.png",
        title="BTAD Sample Defects",
        max_rows=6,
    )


def generate_ksdd2_panel():
    csv_path = Path(r"D:\patchcore\outputs\ksdd2_exploration\ksdd2_defect_size_per_image.csv")

    if not csv_path.exists():
        print("[SKIP] KSDD2 CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    # KSDD2 is single-category surface defect dataset.
    # Select a few train/test anomaly samples.
    if "split" in df.columns:
        df_sample = sample_evenly_by_group(df, group_col="split", n_per_group=3, max_total=3)
    else:
        df_sample = df.head(3)

    records = build_records(df_sample, title_cols=["split", "defect_type"])

    make_original_mask_overlay_panel(
        records=records,
        save_path=OUT_DIR / "ksdd2_sample_defects_panel.png",
        title="KolektorSDD2 Sample Defects",
        max_rows=3,
    )


def generate_loco_panels():
    csv_path = Path(r"D:\patchcore\outputs\loco_exploration\loco_defect_size_per_image.csv")

    if not csv_path.exists():
        print("[SKIP] LOCO CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    if "anomaly_type" not in df.columns:
        print("[SKIP] LOCO CSV does not contain anomaly_type column.")
        return

    # Logical anomalies
    logical_df = df[df["anomaly_type"] == "logical_anomalies"].copy()

    if not logical_df.empty:
        logical_sample = sample_evenly_by_group(
            logical_df,
            group_col="category",
            n_per_group=1,
            max_total=5,
        )

        records = build_records(logical_sample, title_cols=["category", "anomaly_type"])

        make_original_mask_overlay_panel(
            records=records,
            save_path=OUT_DIR / "loco_logical_anomalies_panel.png",
            title="MVTec LOCO AD Logical Anomalies",
            max_rows=5,
        )

    # Structural anomalies
    structural_df = df[df["anomaly_type"] == "structural_anomalies"].copy()

    if not structural_df.empty:
        structural_sample = sample_evenly_by_group(
            structural_df,
            group_col="category",
            n_per_group=1,
            max_total=5,
        )

        records = build_records(structural_sample, title_cols=["category", "anomaly_type"])

        make_original_mask_overlay_panel(
            records=records,
            save_path=OUT_DIR / "loco_structural_anomalies_panel.png",
            title="MVTec LOCO AD Structural Anomalies",
            max_rows=5,
        )


def generate_visa_panel():
    csv_path = Path(r"D:\patchcore\outputs\visa_exploration\visa_defect_size_per_image.csv")

    if not csv_path.exists():
        print("[SKIP] VisA CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    df_sample = sample_evenly_by_group(df, group_col="category", n_per_group=1, max_total=6)

    records = build_records(df_sample, title_cols=["category", "label"])

    make_original_mask_overlay_panel(
        records=records,
        save_path=OUT_DIR / "visa_sample_defects_panel.png",
        title="VisA Sample Defects",
        max_rows=6,
    )







def generate_mvtec_panel():
    csv_path = Path(r"D:\patchcore\outputs\MVTec AD_exploration\mvtec_defect_size_per_image.csv")

    if not csv_path.exists():
        print("[SKIP] MVTec AD CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = fix_mvtec_paths(df)

    # 优先选几个比较有代表性的类别
    preferred_categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "pill",
        "screw",
        "zipper",
    ]

    available_categories = set(df["category"].astype(str).unique().tolist())

    chosen_categories = [c for c in preferred_categories if c in available_categories]

    # 如果 preferred_categories 不够，就自动补别的类别
    if len(chosen_categories) < 3:
        for c in sorted(available_categories):
            if c not in chosen_categories:
                chosen_categories.append(c)
            if len(chosen_categories) >= 3:
                break

    df_sample = df[df["category"].isin(chosen_categories)].copy()

    # 每个类别取 1 张图，最多 3 张
    df_sample = sample_evenly_by_group(
        df_sample,
        group_col="category",
        n_per_group=1,
        max_total=3,
    )

    records = build_records(df_sample, title_cols=["category", "defect_type"])

    make_original_mask_overlay_panel(
        records=records,
        save_path=OUT_DIR / "mvtec_ad_sample_defects_panel.png",
        title="MVTec AD Sample Defects",
        max_rows=3,
    )














def main():
    print("===== Generating dataset sample visualization panels =====")

    generate_btad_panel()
    generate_ksdd2_panel()
    generate_loco_panels()
    generate_visa_panel()
    generate_mvtec_panel()
    print("\n===== Done: sample visualization panels generated =====")
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
