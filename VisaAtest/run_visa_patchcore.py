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

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from visa_dataset import CLASSNAMES, DatasetSplit, VisaDataset


LOGGER = logging.getLogger("visa_patchcore")


def mib(value):
    return round(value / 1024 / 1024, 2)


def safe_minmax(values):
    values = np.asarray(values, dtype=np.float32)
    low = values.min()
    high = values.max()
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore on VisA 1cls.csv.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "VisA"))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--category", default="all", help="VisA class or all")
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


def save_visuals(run_folder, test_dataset, segmentations, scores, max_visuals):
    if max_visuals is None or max_visuals <= 0:
        return []
    image_paths = [row[2] for row in test_dataset.data_to_iterate]
    mask_paths = [row[3] for row in test_dataset.data_to_iterate]
    normal_indices = [
        i for i, path in enumerate(image_paths) if f"{os.sep}Normal{os.sep}" in os.path.normpath(path)
    ]
    anomaly_indices = [i for i in range(len(image_paths)) if i not in normal_indices]
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

    output = run_folder / "segmentation_images"
    output.mkdir(parents=True, exist_ok=True)
    patchcore.utils.plot_segmentation_images(
        str(output),
        image_paths,
        segmentations,
        scores,
        mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    return [str(output / Path(path).name) for path in image_paths]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    run_name = "benchmark_visa_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_folder = output_root / run_name
    run_folder.mkdir(parents=True, exist_ok=False)

    categories = CLASSNAMES if args.category.lower() == "all" else [args.category]
    unknown = sorted(set(categories) - set(CLASSNAMES))
    if unknown:
        raise ValueError("Unknown VisA categories: {}".format(unknown))

    patchcore.utils.fix_seeds(args.seed, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = patchcore.utils.set_torch_device([args.gpu])
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

    LOGGER.info("VisA root: %s", data_root)
    LOGGER.info("Output folder: %s", run_folder)
    LOGGER.info("Device: %s", device)

    for index, category in enumerate(categories, 1):
        LOGGER.info("[%s/%s] Running VisA category %s", index, len(categories), category)
        train_dataset = VisaDataset(
            data_root,
            classname=category,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TRAIN,
        )
        test_dataset = VisaDataset(
            data_root,
            classname=category,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TEST,
        )
        train_loader = make_loader(train_dataset, args)
        test_loader = make_loader(test_dataset, args)
        train_loader.name = "visa_{}".format(category)
        test_loader.name = "visa_{}".format(category)

        with device_context:
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

            scores = safe_minmax(scores)
            segmentations = np.asarray(segmentations, dtype=np.float32)
            segmentations = safe_minmax(segmentations.reshape(len(segmentations), -1)).reshape(
                segmentations.shape
            )

            image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]
            full_pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )["auroc"]
            anomaly_indices = [i for i, mask in enumerate(masks_gt) if np.sum(mask) > 0]
            anomaly_pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in anomaly_indices],
                [masks_gt[i] for i in anomaly_indices],
            )["auroc"]

            model_folder = run_folder / "models" / "visa_{}".format(category)
            model_folder.mkdir(parents=True, exist_ok=True)
            model.save_to_path(str(model_folder))
            kept_visuals.extend(
                save_visuals(
                    run_folder / "visuals" / "visa_{}".format(category),
                    test_dataset,
                    segmentations,
                    scores,
                    args.max_visuals,
                )
            )

            train_memory = process.memory_info().rss
            benchmark_rows.append(
                {
                    "phase": "training",
                    "dataset": "visa_{}".format(category),
                    "num_images": len(train_dataset),
                    "elapsed_seconds": round(train_elapsed, 3),
                    "rss_end_mib": mib(train_memory),
                }
            )
            benchmark_rows.append(
                {
                    "phase": "inference",
                    "dataset": "visa_{}".format(category),
                    "num_images": len(test_dataset),
                    "elapsed_seconds": round(infer_elapsed, 3),
                    "seconds_per_image": round(infer_elapsed / len(test_dataset), 6),
                }
            )
            results.append(
                {
                    "dataset_name": "visa_{}".format(category),
                    "instance_auroc": float(image_auroc),
                    "full_pixel_auroc": float(full_pixel_auroc),
                    "anomaly_pixel_auroc": float(anomaly_pixel_auroc),
                    "train_images": len(train_dataset),
                    "test_normal_images": sum(x[1] == "good" for x in test_dataset.data_to_iterate),
                    "test_anomaly_images": sum(x[1] != "good" for x in test_dataset.data_to_iterate),
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
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - total_start
    metrics = {
        "dataset": "VisA",
        "data_root": str(data_root),
        "split_csv": str(data_root / "split_csv" / "1cls.csv"),
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
        "metrics_note": "This runner reports the same three AUROC metrics as the existing local MVTec benchmark; PRO is not included.",
        "results": results,
        "benchmark_rows": benchmark_rows,
        "kept_visuals": kept_visuals,
    }
    with (run_folder / "benchmark_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with (run_folder / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "dataset_name",
            "instance_auroc",
            "full_pixel_auroc",
            "anomaly_pixel_auroc",
            "train_images",
            "test_normal_images",
            "test_anomaly_images",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow(
            {
                "dataset_name": "Mean",
                "instance_auroc": np.mean([row["instance_auroc"] for row in results]),
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
        file.write("# VisA PatchCore Local Benchmark\n\n")
        file.write("- Data root: `{}`\n".format(data_root))
        file.write("- Split CSV: `{}`\n".format(data_root / "split_csv" / "1cls.csv"))
        file.write("- Categories: `{}`\n".format(", ".join(categories)))
        file.write("- Device: `{}`\n".format(device))
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
                    row["instance_auroc"],
                    row["full_pixel_auroc"],
                    row["anomaly_pixel_auroc"],
                )
            )
        file.write("\nPRO is not included; this matches the existing local MVTec benchmark metrics.\n")

    print("Run folder:", run_folder)
    print("Results:", run_folder / "results.csv")
    print("Report:", run_folder / "benchmark_report.md")


if __name__ == "__main__":
    main()
