import argparse
import contextlib
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
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

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from PIL import Image

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from mvtec_loco_dataset import ANOMALY_TYPES, CLASSNAMES, DatasetSplit, MVTecLoCoDataset


LOGGER = logging.getLogger("mvtec_loco_patchcore")


def mib(value):
    return round(float(value) / 1024 / 1024, 2)


class GPUMonitor:
    """Low-overhead nvidia-smi sampler for utilization and VRAM telemetry."""

    def __init__(self, gpu_index, interval=1.0):
        self.gpu_index = int(gpu_index)
        self.interval = float(interval)
        self.available = shutil.which("nvidia-smi") is not None
        self.samples = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        if not self.available:
            return
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
            line = result.stdout.strip().splitlines()[0]
            utilization, memory_used, memory_total = [
                float(value.strip()) for value in line.split(",")
            ]
            sample = {
                "time": time.perf_counter(),
                "utilization_percent": utilization,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
            with self.lock:
                self.samples.append(sample)
        except (OSError, IndexError, ValueError, subprocess.SubprocessError):
            return

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def mark(self):
        with self.lock:
            return len(self.samples)

    def stats_since(self, start_index):
        with self.lock:
            samples = list(self.samples[start_index:])
        return self._summarize(samples)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def summary(self):
        with self.lock:
            samples = list(self.samples)
        summary = self._summarize(samples)
        summary["available"] = bool(self.available)
        return summary

    @staticmethod
    def _summarize(samples):
        if not samples:
            return {
                "sample_count": 0,
                "average_utilization_percent": None,
                "peak_utilization_percent": None,
                "average_memory_used_mib": None,
                "peak_memory_used_mib": None,
                "memory_total_mib": None,
            }
        utilizations = [sample["utilization_percent"] for sample in samples]
        memory_used = [sample["memory_used_mib"] for sample in samples]
        return {
            "sample_count": len(samples),
            "average_utilization_percent": round(float(np.mean(utilizations)), 2),
            "peak_utilization_percent": round(float(np.max(utilizations)), 2),
            "average_memory_used_mib": round(float(np.mean(memory_used)), 2),
            "peak_memory_used_mib": round(float(np.max(memory_used)), 2),
            "memory_total_mib": round(float(samples[-1]["memory_total_mib"]), 2),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore on MVTec LOCO AD.")
    parser.add_argument(
        "--data-root", default=str(ROOT / "data" / "mvtec_loco_anomaly_detection")
    )
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--category", default="all", help="LoCo category or all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument(
        "--anomaly-type",
        choices=["structural", "logical", "both"],
        default="structural",
        help="Test subset used for the formal benchmark.",
    )
    parser.add_argument(
        "--threshold-method",
        choices=["validation_max", "validation_p99"],
        default="validation_max",
        help="Threshold rule using only normal validation scores.",
    )
    parser.add_argument("--faiss-on-gpu", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
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


def safe_auroc_image(scores, labels):
    if len(set(int(value) for value in labels)) < 2:
        return None
    return float(
        patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels)["auroc"]
    )


def safe_auroc_pixel(segmentations, masks):
    if not segmentations or not masks:
        return None
    flattened = np.concatenate([np.asarray(mask).reshape(-1) for mask in masks])
    if len(np.unique(flattened)) < 2:
        return None
    return float(
        patchcore.metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks
        )["auroc"]
    )


def metric_bundle(scores, segmentations, labels, masks, anomalies):
    scores = np.asarray(scores, dtype=np.float32)
    segmentations = np.asarray(segmentations, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.float32)
    metrics = {
        "image_auroc": safe_auroc_image(scores, labels.tolist()),
        "full_pixel_auroc": safe_auroc_pixel(
            segmentations.tolist(), masks.tolist()
        ),
    }
    for anomaly_type in sorted(set(value for value in anomalies if value != "good")):
        indices = [i for i, value in enumerate(anomalies) if value == anomaly_type]
        combined = [i for i, value in enumerate(anomalies) if value in ("good", anomaly_type)]
        metrics["{}_image_auroc".format(anomaly_type[:-10])] = safe_auroc_image(
            scores[combined].tolist(), labels[combined].tolist()
        )
        metrics["{}_pixel_auroc".format(anomaly_type[:-10])] = safe_auroc_pixel(
            segmentations[indices].tolist(), masks[indices].tolist()
        )
    anomaly_indices = [i for i, mask in enumerate(masks) if np.sum(mask) > 0]
    metrics["anomaly_only_pixel_auroc"] = safe_auroc_pixel(
        segmentations[anomaly_indices].tolist(), masks[anomaly_indices].tolist()
    )
    return metrics


def validation_threshold(scores, method):
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("Validation produced no finite image scores.")
    if method == "validation_p99":
        return float(np.percentile(scores, 99.0))
    return float(np.nextafter(np.max(scores), np.inf))


def threshold_metrics(test_scores, test_labels, validation_scores, threshold):
    test_scores = np.asarray(test_scores, dtype=np.float32)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    validation_scores = np.asarray(validation_scores, dtype=np.float32)
    predicted = test_scores > threshold
    positive = test_labels == 1
    negative = test_labels == 0
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & negative))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "validation_max_score": float(np.max(validation_scores)),
        "validation_mean_score": float(np.mean(validation_scores)),
        "validation_false_positive_rate": float(np.mean(validation_scores > threshold)),
        "test_true_positive": tp,
        "test_true_negative": tn,
        "test_false_positive": fp,
        "test_false_negative": fn,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_accuracy": float(np.mean(predicted == positive)),
    }


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
    normal = [i for i, row in enumerate(test_dataset.data_to_iterate) if row[1] == "good"]
    anomaly = [i for i, row in enumerate(test_dataset.data_to_iterate) if row[1] != "good"]
    selected = (normal[:1] + anomaly)[:max_visuals]
    output = run_folder / "visuals" / test_dataset.classname
    output.mkdir(parents=True, exist_ok=True)
    saved = []

    for index in selected:
        image_path = Path(test_dataset.data_to_iterate[index][2])
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = test_dataset[index]["mask"].squeeze(0).numpy()
        segmentation = np.asarray(segmentations[index])
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground truth")
        axes[2].imshow(image)
        axes[2].imshow(segmentation, cmap="jet", alpha=0.45)
        axes[2].set_title("Prediction {:.3f}".format(float(scores[index])))
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        output_path = output / "{}_{}.png".format(
            test_dataset.data_to_iterate[index][1], image_path.stem
        )
        figure.savefig(output_path, format="png", dpi=120)
        plt.close(figure)
        saved.append(str(output_path))
    return saved


def save_predictions(run_folder, dataset, scores, segmentations, threshold):
    output_path = run_folder / "predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["category", "anomaly_type", "image_path", "image_score", "prediction_map_max", "threshold", "prediction", "label"]
        )
        for row, score, segmentation in zip(dataset.data_to_iterate, scores, segmentations):
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    float(score),
                    float(np.max(segmentation)),
                    float(threshold),
                    int(float(score) > threshold),
                    int(row[1] != "good"),
                ]
            )
    return output_path


def cuda_memory_summary(device):
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    device_index = int(device.index if device.index is not None else 0)
    with torch.cuda.device(device_index):
        return {
            "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
            "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        }


def reset_cuda_peak(device_index):
    if torch.cuda.is_available():
        with torch.cuda.device(device_index):
            torch.cuda.reset_peak_memory_stats()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    if args.anomaly_type == "structural":
        selected_test_anomaly_types = ["structural_anomalies"]
    elif args.anomaly_type == "logical":
        selected_test_anomaly_types = ["logical_anomalies"]
    else:
        selected_test_anomaly_types = list(ANOMALY_TYPES)
    run_name = "benchmark_mvtec_loco_{}_{}".format(
        args.anomaly_type, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_folder = output_root / run_name
    run_folder.mkdir(parents=True, exist_ok=False)

    categories = CLASSNAMES if args.category.lower() == "all" else [args.category]
    unknown = sorted(set(categories) - set(CLASSNAMES))
    if unknown:
        raise ValueError("Unknown MVTec LoCo categories: {}".format(unknown))

    device = patchcore.utils.set_torch_device([args.gpu])
    patchcore.utils.fix_seeds(args.seed, device)
    cuda_device_index = int(device.index if device.index is not None else 0)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    process = psutil.Process(os.getpid())
    results = []
    benchmark_rows = []
    total_start = time.perf_counter()
    gpu_monitor = GPUMonitor(args.gpu, args.gpu_sample_interval)
    gpu_monitor.start()
    reset_cuda_peak(cuda_device_index)
    global_cuda_peak = {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}

    LOGGER.info("MVTec LoCo root: %s", data_root)
    LOGGER.info("Output folder: %s", run_folder)
    LOGGER.info("Device: %s", device)

    for index, category in enumerate(categories, 1):
        LOGGER.info("[%s/%s] Running MVTec LoCo category %s", index, len(categories), category)
        category_gpu_start = gpu_monitor.mark()
        train_dataset = MVTecLoCoDataset(
            data_root, classname=category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TRAIN
        )
        validation_dataset = MVTecLoCoDataset(
            data_root, classname=category, resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.VAL
        )
        test_dataset = MVTecLoCoDataset(
            data_root,
            classname=category,
            resize=args.resize,
            imagesize=args.imagesize,
            split=DatasetSplit.TEST,
            test_anomaly_types=selected_test_anomaly_types,
        )
        train_loader = make_loader(train_dataset, args)
        validation_loader = make_loader(validation_dataset, args)
        test_loader = make_loader(test_dataset, args)
        train_loader.name = "mvtec_loco_{}".format(category)
        validation_loader.name = "mvtec_loco_{}_validation".format(category)
        test_loader.name = "mvtec_loco_{}".format(category)

        with device_context:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reset_cuda_peak(cuda_device_index)
            model = make_patchcore(device, args)

            train_start = time.perf_counter()
            model.fit(train_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_elapsed = time.perf_counter() - train_start
            train_memory_stats = cuda_memory_summary(device)
            for key in global_cuda_peak:
                if train_memory_stats[key] is not None:
                    global_cuda_peak[key] = max(global_cuda_peak[key], train_memory_stats[key])

            reset_cuda_peak(cuda_device_index)
            validation_start = time.perf_counter()
            validation_scores, _, _, _ = model.predict(validation_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            validation_elapsed = time.perf_counter() - validation_start
            validation_memory_stats = cuda_memory_summary(device)
            for key in global_cuda_peak:
                if validation_memory_stats[key] is not None:
                    global_cuda_peak[key] = max(global_cuda_peak[key], validation_memory_stats[key])

            threshold = validation_threshold(validation_scores, args.threshold_method)

            reset_cuda_peak(cuda_device_index)
            infer_start = time.perf_counter()
            scores, segmentations, labels_gt, masks_gt = model.predict(test_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_elapsed = time.perf_counter() - infer_start
            infer_memory_stats = cuda_memory_summary(device)
            for key in global_cuda_peak:
                if infer_memory_stats[key] is not None:
                    global_cuda_peak[key] = max(global_cuda_peak[key], infer_memory_stats[key])

            anomaly_names = [row[1] for row in test_dataset.data_to_iterate]
            category_metrics = metric_bundle(
                scores, segmentations, labels_gt, masks_gt, anomaly_names
            )
            category_threshold_metrics = threshold_metrics(
                scores, labels_gt, validation_scores, threshold
            )
            model_folder = run_folder / "models" / category
            model_folder.mkdir(parents=True, exist_ok=True)
            model.save_to_path(str(model_folder))

            visual_segmentations = normalized_array(
                np.asarray(segmentations).reshape(len(segmentations), -1)
            ).reshape(np.asarray(segmentations).shape)
            visual_scores = normalized_array(scores)
            kept = save_visuals(
                run_folder, test_dataset, visual_segmentations, visual_scores, args.max_visuals
            )
            save_predictions(run_folder, test_dataset, scores, segmentations, threshold)

        category_gpu = gpu_monitor.stats_since(category_gpu_start)
        rss_mib = round(process.memory_info().rss / 1024 / 1024, 2)
        results.append(
            {
                "category": category,
                "train_images": len(train_dataset),
                "validation_images": len(validation_dataset),
                "test_images": len(test_dataset),
                "test_good_images": sum(row[1] == "good" for row in test_dataset.data_to_iterate),
                "test_logical_anomalies": sum(row[1] == "logical_anomalies" for row in test_dataset.data_to_iterate),
                "test_structural_anomalies": sum(row[1] == "structural_anomalies" for row in test_dataset.data_to_iterate),
                "train_seconds": train_elapsed,
                "validation_seconds": validation_elapsed,
                "inference_seconds": infer_elapsed,
                "inference_seconds_per_image": infer_elapsed / len(test_dataset),
                "rss_end_mib": rss_mib,
                "cuda_training_peak_allocated_mib": train_memory_stats["peak_allocated_mib"],
                "cuda_training_peak_reserved_mib": train_memory_stats["peak_reserved_mib"],
                "cuda_validation_peak_allocated_mib": validation_memory_stats["peak_allocated_mib"],
                "cuda_validation_peak_reserved_mib": validation_memory_stats["peak_reserved_mib"],
                "cuda_inference_peak_allocated_mib": infer_memory_stats["peak_allocated_mib"],
                "cuda_inference_peak_reserved_mib": infer_memory_stats["peak_reserved_mib"],
                "gpu_average_utilization_percent": category_gpu["average_utilization_percent"],
                "gpu_peak_utilization_percent": category_gpu["peak_utilization_percent"],
                "gpu_peak_memory_used_mib": category_gpu["peak_memory_used_mib"],
                "gpu_sample_count": category_gpu["sample_count"],
                **category_metrics,
                **category_threshold_metrics,
            }
        )
        benchmark_rows.extend(
            [
                {"category": category, "phase": "training", "images": len(train_dataset), "seconds": train_elapsed, "seconds_per_image": train_elapsed / len(train_dataset), "rss_end_mib": rss_mib, "cuda_peak_allocated_mib": train_memory_stats["peak_allocated_mib"], "cuda_peak_reserved_mib": train_memory_stats["peak_reserved_mib"]},
                {"category": category, "phase": "validation", "images": len(validation_dataset), "seconds": validation_elapsed, "seconds_per_image": validation_elapsed / len(validation_dataset), "rss_end_mib": rss_mib, "cuda_peak_allocated_mib": validation_memory_stats["peak_allocated_mib"], "cuda_peak_reserved_mib": validation_memory_stats["peak_reserved_mib"], "threshold": threshold},
                {"category": category, "phase": "inference", "images": len(test_dataset), "seconds": infer_elapsed, "seconds_per_image": infer_elapsed / len(test_dataset), "rss_end_mib": rss_mib, "gpu_average_utilization_percent": category_gpu["average_utilization_percent"], "gpu_peak_utilization_percent": category_gpu["peak_utilization_percent"], "gpu_peak_memory_used_mib": category_gpu["peak_memory_used_mib"], "cuda_peak_allocated_mib": infer_memory_stats["peak_allocated_mib"], "cuda_peak_reserved_mib": infer_memory_stats["peak_reserved_mib"]},
            ]
        )
        LOGGER.info("Completed %s: %s; threshold=%s", category, category_metrics, threshold)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpu_monitor.stop()
    total_elapsed = time.perf_counter() - total_start
    gpu_summary = gpu_monitor.summary()
    total_train = sum(row["train_seconds"] for row in results)
    total_validation = sum(row["validation_seconds"] for row in results)
    total_inference = sum(row["inference_seconds"] for row in results)
    total_images = sum(row["test_images"] for row in results)
    metrics = {
        "dataset": "MVTec LOCO AD",
        "formal_anomaly_scope": args.anomaly_type,
        "formal_test_anomaly_types": selected_test_anomaly_types,
        "logical_anomalies_in_formal_metrics": False,
        "data_root": str(data_root),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "categories": categories,
        "arguments": vars(args),
        "results": results,
        "total_elapsed_seconds": total_elapsed,
        "total_train_seconds": total_train,
        "total_validation_seconds": total_validation,
        "total_inference_seconds": total_inference,
        "weighted_average_inference_seconds_per_image": total_inference / total_images,
        "gpu": {
            "gpu_index": args.gpu,
            "utilization_definition": "nvidia-smi GPU compute utilization sampled during the benchmark",
            "memory_definition": "nvidia-smi framebuffer memory used; CUDA values are process allocator peaks",
            **gpu_summary,
            "peak_allocated_mib": global_cuda_peak["peak_allocated_mib"],
            "peak_reserved_mib": global_cuda_peak["peak_reserved_mib"],
        },
    }
    with (run_folder / "benchmark_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    summary_fields = sorted({key for row in benchmark_rows for key in row})
    with (run_folder / "benchmark_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    report_lines = [
        "MVTec LOCO AD PatchCore benchmark",
        "run_folder: {}".format(run_folder),
        "device: {}".format(device),
        "categories: {}".format(", ".join(categories)),
        "total_elapsed_seconds: {:.3f}".format(total_elapsed),
        "total_train_seconds: {:.3f}".format(total_train),
        "total_validation_seconds: {:.3f}".format(total_validation),
        "total_inference_seconds: {:.3f}".format(total_inference),
        "weighted_average_inference_seconds_per_image: {:.6f}".format(total_inference / total_images),
        "gpu_average_utilization_percent: {}".format(gpu_summary["average_utilization_percent"]),
        "gpu_peak_utilization_percent: {}".format(gpu_summary["peak_utilization_percent"]),
        "gpu_average_memory_used_mib: {}".format(gpu_summary["average_memory_used_mib"]),
        "gpu_peak_memory_used_mib: {}".format(gpu_summary["peak_memory_used_mib"]),
        "cuda_peak_allocated_mib: {}".format(global_cuda_peak["peak_allocated_mib"]),
        "cuda_peak_reserved_mib: {}".format(global_cuda_peak["peak_reserved_mib"]),
        "gpu_sample_count: {}".format(gpu_summary["sample_count"]),
        "",
    ]
    for result in results:
        report_lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    (run_folder / "benchmark_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    LOGGER.info("Benchmark complete: %s", run_folder)


if __name__ == "__main__":
    main()
