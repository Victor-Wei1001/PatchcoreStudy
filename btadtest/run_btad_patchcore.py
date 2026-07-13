import argparse
import contextlib
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PATCHCORE_SRC = ROOT / "patchcore-inspection-main" / "src"
if str(PATCHCORE_SRC) not in sys.path:
    sys.path.insert(0, str(PATCHCORE_SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psutil
import torch
import matplotlib.pyplot as plt
from PIL import Image

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from btad_dataset import CLASSNAMES, BTADDataset, DatasetSplit


LOGGER = logging.getLogger("btad_patchcore")


def mib(value):
    return round(value / 1024 / 1024, 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore on BTAD.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "btad"))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--category", default="all", help="BTAD category or all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--faiss-on-gpu", action="store_true")
    return parser.parse_args()


def make_patchcore(device, args):
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(args.coreset, device)
    nn_method = patchcore.common.FaissNN(args.faiss_on_gpu, 4)
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=sampler,
        anomaly_scorer_num_nn=1,
        nn_method=nn_method,
    )
    return model


def make_loader(dataset, args):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def normalized_array(values):
    values = np.asarray(values, dtype=np.float32)
    low = values.min()
    high = values.max()
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def save_visuals(run_folder, test_dataset, segmentations, scores, max_visuals):
    if max_visuals is None or max_visuals <= 0:
        return []

    image_paths = [row[2] for row in test_dataset.data_to_iterate]
    mask_paths = [row[3] for row in test_dataset.data_to_iterate]
    normal_indices = [
        i for i, row in enumerate(test_dataset.data_to_iterate) if row[1] == "good"
    ]
    anomaly_indices = [
        i for i, row in enumerate(test_dataset.data_to_iterate) if row[1] != "good"
    ]
    selected = (normal_indices[:1] + anomaly_indices)[:max_visuals]

    image_paths = [image_paths[i] for i in selected]
    mask_paths = [mask_paths[i] for i in selected]
    segmentations = [segmentations[i] for i in selected]
    scores = [scores[i] for i in selected]

    def image_transform(image):
        image = test_dataset.transform_img(image)
        mean = np.asarray(test_dataset.transform_mean).reshape(-1, 1, 1)
        std = np.asarray(test_dataset.transform_std).reshape(-1, 1, 1)
        return np.clip((image.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)

    def mask_transform(mask):
        mask = mask.convert("L")
        return (test_dataset.transform_mask(mask).numpy() > 0).astype(np.float32)

    # The shared PatchCore visualizer preserves the source suffix. BTAD uses
    # BMP images, but Matplotlib cannot save figures with a .bmp suffix, so
    # keep this small BTAD-local visualizer and always write PNG files.
    output = run_folder / "segmentation_images"
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for image_path, mask_path, score, segmentation in zip(
        image_paths, mask_paths, scores, segmentations
    ):
        image = image_transform(Image.open(image_path).convert("RGB"))
        if not isinstance(image, np.ndarray):
            image = image.numpy()
        if mask_path is None:
            mask = np.zeros_like(image)
        else:
            mask = mask_transform(Image.open(mask_path).convert("RGB"))
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()
        figure, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[0].set_title("Image")
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[1].set_title("Ground truth")
        axes[2].imshow(segmentation)
        axes[2].set_title("Prediction {:.3f}".format(float(score)))
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        source_path = Path(image_path)
        output_path = output / "{}_{}.png".format(source_path.parent.name, source_path.stem)
        figure.savefig(output_path, format="png", dpi=120)
        plt.close(figure)
        saved.append(str(output_path))
    return saved


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    run_name = "benchmark_btad_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_folder = output_root / run_name
    run_folder.mkdir(parents=True, exist_ok=False)

    categories = CLASSNAMES if args.category.lower() == "all" else [args.category]
    unknown = sorted(set(categories) - set(CLASSNAMES))
    if unknown:
        raise ValueError("Unknown BTAD categories: {}".format(unknown))

    device = patchcore.utils.set_torch_device([args.gpu])
    patchcore.utils.fix_seeds(args.seed, device)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    process = psutil.Process(os.getpid())
    results = []
    benchmark_rows = []
    kept_visuals = []
    total_start = time.perf_counter()

    LOGGER.info("BTAD root: %s", data_root)
    LOGGER.info("Output folder: %s", run_folder)
    LOGGER.info("Device: %s", device)

    for index, category in enumerate(categories, 1):
        LOGGER.info("[%s/%s] Running BTAD category %s", index, len(categories), category)
        train_dataset = BTADDataset(
            data_root,
            classname=category,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TRAIN,
        )
        test_dataset = BTADDataset(
            data_root,
            classname=category,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TEST,
        )
        train_loader = make_loader(train_dataset, args)
        test_loader = make_loader(test_dataset, args)
        train_loader.name = "btad_{}".format(category)
        test_loader.name = "btad_{}".format(category)

        with device_context:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = make_patchcore(device, args)

            train_start = time.perf_counter()
            model.fit(train_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_elapsed = time.perf_counter() - train_start

            infer_start = time.perf_counter()
            scores, segmentations, labels_gt, masks_gt = model.predict(test_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_elapsed = time.perf_counter() - infer_start

            raw_scores = np.asarray(scores, dtype=np.float32)
            raw_segmentations = np.asarray(segmentations, dtype=np.float32)
            image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                raw_scores, labels_gt
            )["auroc"]
            full_pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                raw_segmentations, masks_gt
            )["auroc"]
            anomaly_indices = [i for i, mask in enumerate(masks_gt) if np.sum(mask) > 0]
            anomaly_pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [raw_segmentations[i] for i in anomaly_indices],
                [masks_gt[i] for i in anomaly_indices],
            )["auroc"]

            model_folder = run_folder / "models" / "btad_{}".format(category)
            model_folder.mkdir(parents=True, exist_ok=True)
            model.save_to_path(str(model_folder))

            visual_segmentations = normalized_array(
                raw_segmentations.reshape(len(raw_segmentations), -1)
            ).reshape(raw_segmentations.shape)
            visual_scores = normalized_array(raw_scores)
            kept_visuals.extend(
                save_visuals(
                    run_folder / "visuals" / "btad_{}".format(category),
                    test_dataset,
                    visual_segmentations,
                    visual_scores,
                    args.max_visuals,
                )
            )

            benchmark_rows.append(
                {
                    "phase": "training",
                    "dataset": "btad_{}".format(category),
                    "num_images": len(train_dataset),
                    "elapsed_seconds": round(train_elapsed, 3),
                    "rss_end_mib": mib(process.memory_info().rss),
                }
            )
            benchmark_rows.append(
                {
                    "phase": "inference",
                    "dataset": "btad_{}".format(category),
                    "num_images": len(test_dataset),
                    "elapsed_seconds": round(infer_elapsed, 3),
                    "seconds_per_image": round(infer_elapsed / len(test_dataset), 6),
                }
            )
            results.append(
                {
                    "dataset_name": "btad_{}".format(category),
                    "image_auroc": float(image_auroc),
                    "full_pixel_auroc": float(full_pixel_auroc),
                    "anomaly_pixel_auroc": float(anomaly_pixel_auroc),
                    "train_images": len(train_dataset),
                    "test_normal_images": sum(row[1] == "good" for row in test_dataset.data_to_iterate),
                    "test_anomaly_images": sum(row[1] != "good" for row in test_dataset.data_to_iterate),
                }
            )
            LOGGER.info(
                "%s: image AUROC=%.4f, full pixel AUROC=%.4f, anomaly pixel AUROC=%.4f",
                category,
                image_auroc,
                full_pixel_auroc,
                anomaly_pixel_auroc,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - total_start
    metrics = {
        "dataset": "BTAD",
        "data_root": str(data_root),
        "categories": categories,
        "seed": args.seed,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "coreset": args.coreset,
        "backbone": "wideresnet50",
        "layers": ["layer2", "layer3"],
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "total_elapsed_seconds": round(elapsed, 3),
        "metrics_note": "Image AUROC, full pixel AUROC and anomaly-only pixel AUROC. PRO is not included.",
        "results": results,
        "benchmark_rows": benchmark_rows,
        "kept_visuals": kept_visuals,
    }
    with (run_folder / "benchmark_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    result_fields = [
        "dataset_name",
        "image_auroc",
        "full_pixel_auroc",
        "anomaly_pixel_auroc",
        "train_images",
        "test_normal_images",
        "test_anomaly_images",
    ]
    with (run_folder / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(
            {
                "dataset_name": "Mean",
                "image_auroc": np.mean([row["image_auroc"] for row in results]),
                "full_pixel_auroc": np.mean([row["full_pixel_auroc"] for row in results]),
                "anomaly_pixel_auroc": np.mean([row["anomaly_pixel_auroc"] for row in results]),
            }
        )

    with (run_folder / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = sorted({key for row in benchmark_rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    with (run_folder / "benchmark_report.md").open("w", encoding="utf-8") as file:
        file.write("# BTAD PatchCore Local Benchmark\n\n")
        file.write("- Data root: `{}`\n".format(data_root))
        file.write("- Categories: `{}`\n".format(", ".join(categories)))
        file.write("- Device: `{}`\n".format(device))
        file.write("- Resize / image size: `{}` / `{}`\n".format(args.resize, args.imagesize))
        file.write("- Total elapsed seconds: `{}`\n\n".format(round(elapsed, 3)))
        file.write("| Category | Train | Test normal | Test anomaly | Image AUROC | Full pixel AUROC | Anomaly pixel AUROC |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in results:
            file.write(
                "| {} | {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |\n".format(
                    row["dataset_name"],
                    row["train_images"],
                    row["test_normal_images"],
                    row["test_anomaly_images"],
                    row["image_auroc"],
                    row["full_pixel_auroc"],
                    row["anomaly_pixel_auroc"],
                )
            )
        file.write("\nPRO is not included in this local benchmark.\n")

    print("Run folder:", run_folder)
    print("Results:", run_folder / "results.csv")
    print("Report:", run_folder / "benchmark_report.md")


if __name__ == "__main__":
    main()
