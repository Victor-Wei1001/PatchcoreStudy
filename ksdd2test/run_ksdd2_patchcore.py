import argparse
import contextlib
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "patchcore-inspection-main" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from patchcore.datasets.ksdd2 import DatasetSplit, KolektorSDD2Dataset
import patchcore.utils
from mvteclocotest.run_mvtec_loco_patchcore import (
    GPUMonitor,
    cuda_memory_summary,
    make_loader,
    make_patchcore,
    metric_bundle,
    mib,
    normalized_array,
    reset_cuda_peak,
    save_visuals,
)


LOGGER = logging.getLogger("ksdd2_patchcore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PatchCore on KolektorSDD2.")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "KolektorSDD2"))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--faiss-on-gpu", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def setup_logging(folder):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(folder / "benchmark_run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[stream, file_handler], force=True)


def save_predictions_unthresholded(run_folder, dataset, scores, segmentations):
    output_path = run_folder / "predictions.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["category", "anomaly_type", "image_path", "image_score", "prediction_map_max", "label"]
        )
        for row, score, segmentation in zip(dataset.data_to_iterate, scores, segmentations):
            writer.writerow(
                [row[0], row[1], row[2], float(score), float(np.max(segmentation)), int(row[1] != "good")]
            )
    return output_path


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    run_name = args.run_name or "benchmark_ksdd2_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_folder = output_root / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    setup_logging(run_folder)

    device = patchcore.utils.set_torch_device([args.gpu])
    patchcore.utils.fix_seeds(args.seed, device)
    device_index = int(device.index if device.index is not None else 0)
    device_context = torch.cuda.device("cuda:{}".format(device.index)) if "cuda" in device.type.lower() else contextlib.suppress()
    process = psutil.Process()
    monitor = GPUMonitor(args.gpu, args.gpu_sample_interval)
    monitor.start()
    total_start = time.perf_counter()
    cuda_peak = {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}

    train = KolektorSDD2Dataset(data_root, classname="Kolektor_surface", resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TRAIN)
    test = KolektorSDD2Dataset(data_root, classname="Kolektor_surface", resize=args.resize, imagesize=args.imagesize, split=DatasetSplit.TEST)
    train_loader, test_loader = make_loader(train, args), make_loader(test, args)
    train_loader.name, test_loader.name = "ksdd2_train_normal", "ksdd2_test"
    LOGGER.info("Dataset: train valid=%d normal=%d defect=%d skipped=%d; memory=%d; test valid=%d normal=%d defect=%d skipped=%d", train.total_images, train.normal_images, train.anomaly_images, train.skipped_images, len(train), test.total_images, test.normal_images, test.anomaly_images, test.skipped_images)

    with device_context:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_cuda_peak(device_index)
        model = make_patchcore(device, args)

        start = time.perf_counter()
        model.fit(train_loader)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_seconds = time.perf_counter() - start
        train_memory = cuda_memory_summary(device)

        reset_cuda_peak(device_index)
        start = time.perf_counter()
        scores, segmentations, labels_gt, masks_gt = model.predict(test_loader)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        inference_memory = cuda_memory_summary(device)
        for memory in (train_memory, inference_memory):
            for key in cuda_peak:
                if memory[key] is not None: cuda_peak[key] = max(cuda_peak[key], memory[key])

        anomalies = [row[1] for row in test.data_to_iterate]
        quality = metric_bundle(scores, segmentations, labels_gt, masks_gt, anomalies)
        # The shared LoCo helper shortens anomaly names by removing the
        # "anomalies" suffix.  SDD2 uses the shorter label "defect", so make
        # the resulting report fields explicit instead of producing a leading
        # underscore.
        if "_image_auroc" in quality:
            quality["defect_image_auroc"] = quality.pop("_image_auroc")
        if "_pixel_auroc" in quality:
            quality["defect_pixel_auroc"] = quality.pop("_pixel_auroc")
        model_folder = run_folder / "models" / "Kolektor_surface"
        model_folder.mkdir(parents=True, exist_ok=True)
        model.save_to_path(str(model_folder))
        visual_maps = normalized_array(np.asarray(segmentations).reshape(len(segmentations), -1)).reshape(np.asarray(segmentations).shape)
        visuals = save_visuals(run_folder, test, visual_maps, normalized_array(scores), args.max_visuals)
        prediction_csv = save_predictions_unthresholded(run_folder, test, scores, segmentations)

    monitor.stop()
    gpu = monitor.summary()
    total_seconds = time.perf_counter() - total_start
    rss = mib(process.memory_info().rss)
    result = {
        "category": "Kolektor_surface",
        "train_valid_images": train.total_images, "train_normal_images": train.normal_images, "train_defect_images": train.anomaly_images, "train_skipped_images": train.skipped_images,
        "training_images_used": len(train),
        "test_valid_images": test.total_images, "test_good_images": test.normal_images, "test_defect_images": test.anomaly_images, "test_skipped_images": test.skipped_images,
        "train_seconds": train_seconds, "inference_seconds": inference_seconds, "inference_seconds_per_image": inference_seconds / len(test),
        "rss_end_mib": rss,
        "cuda_training_peak_allocated_mib": train_memory["peak_allocated_mib"], "cuda_training_peak_reserved_mib": train_memory["peak_reserved_mib"],
        "cuda_inference_peak_allocated_mib": inference_memory["peak_allocated_mib"], "cuda_inference_peak_reserved_mib": inference_memory["peak_reserved_mib"],
        "gpu_average_utilization_percent": gpu["average_utilization_percent"], "gpu_peak_utilization_percent": gpu["peak_utilization_percent"], "gpu_peak_memory_used_mib": gpu["peak_memory_used_mib"], "gpu_sample_count": gpu["sample_count"],
        **quality,
    }
    rows = [
        {"category": "Kolektor_surface", "phase": "training", "images": len(train), "seconds": train_seconds, "seconds_per_image": train_seconds / len(train), "rss_end_mib": rss, "cuda_peak_allocated_mib": train_memory["peak_allocated_mib"], "cuda_peak_reserved_mib": train_memory["peak_reserved_mib"]},
        {"category": "Kolektor_surface", "phase": "inference", "images": len(test), "seconds": inference_seconds, "seconds_per_image": inference_seconds / len(test), "rss_end_mib": rss, "gpu_average_utilization_percent": gpu["average_utilization_percent"], "gpu_peak_utilization_percent": gpu["peak_utilization_percent"], "gpu_peak_memory_used_mib": gpu["peak_memory_used_mib"], "cuda_peak_allocated_mib": inference_memory["peak_allocated_mib"], "cuda_peak_reserved_mib": inference_memory["peak_reserved_mib"]},
    ]
    metrics = {
        "dataset": "KolektorSDD2", "data_root": str(data_root), "device": str(device), "python_executable": sys.executable, "python_version": sys.version, "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else None,
        "arguments": vars(args), "results": [result], "benchmark_rows": rows, "total_elapsed_seconds": total_seconds, "total_train_seconds": train_seconds, "total_inference_seconds": inference_seconds, "weighted_average_inference_seconds_per_image": inference_seconds / len(test), "gpu": {"gpu_index": args.gpu, "utilization_definition": "nvidia-smi GPU compute utilization sampled during the benchmark", "memory_definition": "nvidia-smi framebuffer memory used; CUDA values are process allocator peaks", **gpu, **cuda_peak}, "predictions_csv": str(prediction_csv), "visuals": visuals,
    }
    (run_folder / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (run_folder / "benchmark_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = ["KolektorSDD2 PatchCore benchmark", "run_folder: {}".format(run_folder), "data_root: {}".format(data_root), "device: {}".format(device), "gpu_name: {}".format(metrics["gpu_name"]), "train_valid_images: {}".format(train.total_images), "train_normal_images: {}".format(train.normal_images), "train_defect_images: {}".format(train.anomaly_images), "train_skipped_images: {}".format(train.skipped_images), "training_images_used: {}".format(len(train)), "test_valid_images: {}".format(test.total_images), "test_good_images: {}".format(test.normal_images), "test_defect_images: {}".format(test.anomaly_images), "test_skipped_images: {}".format(test.skipped_images), "total_elapsed_seconds: {:.3f}".format(total_seconds), "total_train_seconds: {:.3f}".format(train_seconds), "total_inference_seconds: {:.3f}".format(inference_seconds), "weighted_average_inference_seconds_per_image: {:.6f}".format(inference_seconds / len(test)), "gpu_average_utilization_percent: {}".format(gpu["average_utilization_percent"]), "gpu_peak_utilization_percent: {}".format(gpu["peak_utilization_percent"]), "gpu_average_memory_used_mib: {}".format(gpu["average_memory_used_mib"]), "gpu_peak_memory_used_mib: {}".format(gpu["peak_memory_used_mib"]), "cuda_peak_allocated_mib: {}".format(cuda_peak["peak_allocated_mib"]), "cuda_peak_reserved_mib: {}".format(cuda_peak["peak_reserved_mib"]), "gpu_sample_count: {}".format(gpu["sample_count"]), "", json.dumps(result, ensure_ascii=False, indent=2)]
    (run_folder / "benchmark_report.txt").write_text("\n".join(report), encoding="utf-8")
    LOGGER.info("Benchmark complete: %s", run_folder)


if __name__ == "__main__":
    main()
