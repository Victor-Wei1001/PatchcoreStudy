from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_image_rgb(image_path):
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    return img


def load_mask_binary(mask_path, target_size=None):
    """
    Load mask and convert it to binary mask.

    Non-zero pixels are treated as defect pixels.
    If target_size is provided, resize mask to match the original image size.
    """
    mask_path = Path(mask_path)
    mask = Image.open(mask_path).convert("L")

    if target_size is not None and mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)

    mask_array = np.array(mask)
    binary_mask = mask_array > 0

    return binary_mask


def compute_bbox_from_mask(binary_mask):
    """
    Compute bounding box from a binary mask.

    Return:
        (x_min, y_min, x_max, y_max)

    If mask is empty, return None.
    """
    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def create_overlay_image(image, binary_mask, alpha=0.45):
    """
    Create red mask overlay on the original image.
    """
    image_array = np.array(image).astype(np.float32)

    overlay = image_array.copy()
    red_layer = np.zeros_like(image_array)
    red_layer[..., 0] = 255

    mask_3d = binary_mask[..., None]

    overlay = np.where(
        mask_3d,
        image_array * (1 - alpha) + red_layer * alpha,
        image_array
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


def draw_bbox(image, bbox, color=(255, 0, 0), width=3):
    """
    Draw bounding box on image.
    """
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        for i in range(width):
            draw.rectangle(
                [x_min - i, y_min - i, x_max + i, y_max + i],
                outline=color
            )

    return image_with_box


def make_mask_display(binary_mask):
    """
    Convert binary mask to black-white image for visualization.
    """
    mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    return Image.fromarray(mask_uint8, mode="L")


def make_original_mask_overlay_panel(records, save_path, title=None, max_rows=5):
    """
    Make a panel like:

    Original | Ground Truth Mask | Overlay + Bounding Box

    records should be a list of dicts:
    {
        "image_path": "...",
        "mask_path": "...",
        "row_title": "..."
    }
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    valid_records = []

    for r in records:
        image_path = Path(r["image_path"])
        mask_path = Path(r["mask_path"])

        if not image_path.exists():
            print(f"[WARNING] Missing image: {image_path}")
            continue

        if not mask_path.exists():
            print(f"[WARNING] Missing mask: {mask_path}")
            continue

        valid_records.append(r)

    if len(valid_records) == 0:
        print(f"[WARNING] No valid records for panel: {save_path}")
        return

    valid_records = valid_records[:max_rows]
    n_rows = len(valid_records)

    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3.2 * n_rows))

    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, record in enumerate(valid_records):
        image_path = Path(record["image_path"])
        mask_path = Path(record["mask_path"])
        row_title = record.get("row_title", image_path.name)

        image = load_image_rgb(image_path)
        binary_mask = load_mask_binary(mask_path, target_size=image.size)

        bbox = compute_bbox_from_mask(binary_mask)

        mask_display = make_mask_display(binary_mask)
        overlay = create_overlay_image(image, binary_mask)
        overlay_box = draw_bbox(overlay, bbox)

        ax_original = axes[row_idx, 0]
        ax_mask = axes[row_idx, 1]
        ax_overlay = axes[row_idx, 2]

        ax_original.imshow(image)
        ax_original.set_title(f"Original: {row_title}", fontsize=8)
        ax_original.axis("off")

        ax_mask.imshow(mask_display, cmap="gray")
        ax_mask.set_title("Ground Truth Mask", fontsize=8)
        ax_mask.axis("off")

        ax_overlay.imshow(overlay_box)
        ax_overlay.set_title("Overlay + Bounding Box", fontsize=8)
        ax_overlay.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved sample panel: {save_path}")