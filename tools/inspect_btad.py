from pathlib import Path

ROOT = Path(r"D:\patchcore\data\BTAD")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

print("===== BTAD Root =====")
print(ROOT)
print("Exists:", ROOT.exists())

print("\n===== Top-level files/folders =====")
if ROOT.exists():
    for p in ROOT.iterdir():
        print(p.name)

print("\n===== First 100 directories =====")
dirs = [p for p in ROOT.rglob("*") if p.is_dir()]
for p in dirs[:100]:
    print(p)

print("\n===== Image-like files count =====")
img_files = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
print("Total image-like files:", len(img_files))

print("\n===== First 60 image-like files =====")
for p in img_files[:60]:
    print(p)