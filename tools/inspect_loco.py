from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\patchcore\data\mvtec_loco_anomaly_detection")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    print("===== MVTec LOCO AD Root =====")
    print(ROOT)
    print("Exists:", ROOT.exists())

    if not ROOT.exists():
        print("Root path does not exist.")
        return

    print("\n===== Top-level files/folders =====")
    for p in ROOT.iterdir():
        print(p.name)

    print("\n===== First 160 directories =====")
    dirs = [p for p in ROOT.rglob("*") if p.is_dir()]
    for p in dirs[:160]:
        print(p)

    print("\n===== Image-like files count =====")
    img_files = [
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    print("Total image-like files:", len(img_files))

    print("\n===== First 100 image-like files =====")
    for p in img_files[:100]:
        print(p)

    print("\n===== Folder name frequency =====")
    folder_names = [p.name.lower() for p in dirs]
    for name, count in Counter(folder_names).most_common(50):
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()