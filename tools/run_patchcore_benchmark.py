import argparse
import csv
import json
import os
import runpy
import shutil
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import certifi
import psutil
import torch

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)


MVTEC_CLASSES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def mib(value):
    return round(value / 1024 / 1024, 2)


def folder_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def newest_run_folder(results_root, project, run_name):
    project_root = Path(results_root) / project
    candidates = [project_root / run_name]
    candidates.extend(sorted(project_root.glob(run_name + "_*")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return project_root / run_name
    return max(existing, key=lambda path: path.stat().st_mtime)


def trim_visuals(run_folder, max_visuals):
    if max_visuals is None or max_visuals < 1:
        return []

    kept = []
    seg_root = Path(run_folder) / "segmentation_images"
    if not seg_root.exists():
        return kept

    for dataset_dir in sorted([path for path in seg_root.iterdir() if path.is_dir()]):
        images = sorted(dataset_dir.glob("*.png"))
        if len(images) <= max_visuals:
            kept.extend(str(path) for path in images)
            continue

        selected = []
        normal = [path for path in images if "_good_" in path.name]
        abnormal = [path for path in images if "_good_" not in path.name]
        if normal:
            selected.append(normal[0])
        selected.extend(abnormal[: max_visuals - len(selected)])
        selected = selected[:max_visuals]

        keep_names = {path.name for path in selected}
        archive = dataset_dir / "_extra_visuals_not_kept"
        archive.mkdir(exist_ok=True)
        for path in images:
            if path.name not in keep_names:
                shutil.move(str(path), str(archive / path.name))
        kept.extend(str(path) for path in selected)

    return kept


def parse_args():
    parser = argparse.ArgumentParser(description="Run official PatchCore and collect local benchmark metrics.")
    parser.add_argument("--repo", default=r"D:\patchcore\patchcore-inspection-main")
    parser.add_argument("--data-root", default=r"D:\patchcore\data\mvtecAD")
    parser.add_argument("--results-root", default=r"D:\patchcore\outputs\patchcore_runs")
    parser.add_argument("--category", default="bottle", help="MVTec category, or all.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--faiss-on-gpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(args.repo)
    src = repo / "src"
    run_script = repo / "bin" / "run_patchcore.py"
    project = "PatchCoreLocal"
    run_name = args.run_name or "benchmark_{}_{}".format(args.category, datetime.now().strftime("%Y%m%d_%H%M%S"))

    sys.path.insert(0, str(src))
    os.environ["PYTHONPATH"] = str(src)

    import patchcore.patchcore as patchcore_module
    import patchcore.utils as patchcore_utils

    process = psutil.Process(os.getpid())
    metrics = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "repo": str(repo),
        "data_root": str(Path(args.data_root)),
        "category": args.category,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "faiss_on_gpu": bool(args.faiss_on_gpu),
        "training": [],
        "inference": [],
        "losses": "PatchCore does not train with backpropagation loss; it builds a normal-feature memory bank from a pretrained backbone.",
    }

    original_fit = patchcore_module.PatchCore.fit
    original_predict = patchcore_module.PatchCore.predict
    original_plot_segmentation_images = patchcore_utils.plot_segmentation_images

    def fit_with_metrics(self, training_data):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rss_start = process.memory_info().rss
        start = time.perf_counter()
        result = original_fit(self, training_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rss_end = process.memory_info().rss
        metrics["training"].append(
            {
                "dataset": getattr(training_data, "name", None),
                "num_images": len(training_data.dataset),
                "elapsed_seconds": round(elapsed, 3),
                "rss_start_mib": mib(rss_start),
                "rss_end_mib": mib(rss_end),
                "rss_delta_mib": mib(rss_end - rss_start),
                "cuda_peak_allocated_mib": mib(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "cuda_peak_reserved_mib": mib(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            }
        )
        return result

    def predict_with_metrics(self, data):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original_predict(self, data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        num_images = len(data.dataset)
        metrics["inference"].append(
            {
                "dataset": getattr(data, "name", None),
                "num_images": num_images,
                "elapsed_seconds": round(elapsed, 3),
                "seconds_per_image": round(elapsed / num_images, 6),
                "cuda_peak_allocated_mib": mib(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "cuda_peak_reserved_mib": mib(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            }
        )
        return result

    patchcore_module.PatchCore.fit = fit_with_metrics
    patchcore_module.PatchCore.predict = predict_with_metrics

    def plot_limited_visuals(
        savefolder,
        image_paths,
        segmentations,
        anomaly_scores=None,
        mask_paths=None,
        image_transform=lambda x: x,
        mask_transform=lambda x: x,
        save_depth=4,
    ):
        if args.max_visuals is not None and args.max_visuals > 0:
            normal = [idx for idx, path in enumerate(image_paths) if "{}good{}".format(os.sep, os.sep) in os.path.normpath(path)]
            abnormal = [idx for idx, path in enumerate(image_paths) if idx not in normal]
            selected = []
            if normal:
                selected.append(normal[0])
            selected.extend(abnormal[: args.max_visuals - len(selected)])
            selected = selected[: args.max_visuals]
            image_paths = [image_paths[idx] for idx in selected]
            segmentations = [segmentations[idx] for idx in selected]
            if anomaly_scores is not None:
                anomaly_scores = [anomaly_scores[idx] for idx in selected]
            if mask_paths is not None:
                mask_paths = [mask_paths[idx] for idx in selected]
        return original_plot_segmentation_images(
            savefolder,
            image_paths,
            segmentations,
            anomaly_scores=anomaly_scores,
            mask_paths=mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
            save_depth=save_depth,
        )

    patchcore_utils.plot_segmentation_images = plot_limited_visuals

    categories = MVTEC_CLASSES if args.category.lower() == "all" else [args.category]
    command = [
        str(run_script),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--save_patchcore_model",
        "--save_segmentation_images",
        "--log_group",
        run_name,
        "--log_project",
        project,
        str(Path(args.results_root)),
        "patch_core",
        "-b",
        "wideresnet50",
        "-le",
        "layer2",
        "-le",
        "layer3",
        "--pretrain_embed_dimension",
        "1024",
        "--target_embed_dimension",
        "1024",
        "--anomaly_scorer_num_nn",
        "1",
        "--patchsize",
        "3",
    ]
    if args.faiss_on_gpu:
        command.append("--faiss_on_gpu")
    command.extend(
        [
            "sampler",
            "-p",
            str(args.coreset),
            "approx_greedy_coreset",
            "dataset",
            "--resize",
            str(args.resize),
            "--imagesize",
            str(args.imagesize),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
    )
    for category in categories:
        command.extend(["-d", category])
    command.extend(["mvtec", str(Path(args.data_root))])

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    total_start = time.perf_counter()
    try:
        os.chdir(repo)
        sys.argv = command
        try:
            runpy.run_path(str(run_script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    run_folder = newest_run_folder(args.results_root, project, run_name)
    kept_visuals = trim_visuals(run_folder, args.max_visuals)
    metrics["ended_at"] = datetime.now().isoformat(timespec="seconds")
    metrics["total_elapsed_seconds"] = round(time.perf_counter() - total_start, 3)
    metrics["run_folder"] = str(run_folder)
    metrics["results_csv"] = str(run_folder / "results.csv")
    metrics["trained_model_total_size_mib"] = mib(folder_size(run_folder / "models"))
    metrics["kept_visuals"] = kept_visuals
    metrics["official_command"] = "python " + " ".join(command)

    report_json = run_folder / "benchmark_metrics.json"
    report_csv = run_folder / "benchmark_summary.csv"
    report_md = run_folder / "benchmark_report.md"

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    rows = [{"phase": "training", **row} for row in metrics["training"]]
    rows.extend({"phase": "inference", **row} for row in metrics["inference"])
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with report_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with report_md.open("w", encoding="utf-8") as file:
        file.write("# PatchCore Local Benchmark\n\n")
        file.write("- Run folder: `{}`\n".format(run_folder))
        file.write("- GPU: `{}`\n".format(metrics["gpu_name"]))
        file.write("- CUDA available: `{}`\n".format(metrics["cuda_available"]))
        file.write("- Total elapsed seconds: `{}`\n".format(metrics["total_elapsed_seconds"]))
        file.write("- Trained model total size MiB: `{}`\n".format(metrics["trained_model_total_size_mib"]))
        file.write("- Losses: `{}`\n".format(metrics["losses"]))
        file.write("- Results CSV: `{}`\n".format(metrics["results_csv"]))
        file.write("\n## Training\n\n")
        for row in metrics["training"]:
            file.write("- `{dataset}`: {elapsed_seconds}s, CUDA peak allocated {cuda_peak_allocated_mib} MiB, RSS delta {rss_delta_mib} MiB\n".format(**row))
        file.write("\n## Inference\n\n")
        for row in metrics["inference"]:
            file.write("- `{dataset}`: {elapsed_seconds}s total, {seconds_per_image}s/image, CUDA peak allocated {cuda_peak_allocated_mib} MiB\n".format(**row))
        file.write("\n## Visuals Kept\n\n")
        for path in kept_visuals:
            file.write("- `{}`\n".format(path))

    print("Run folder:", run_folder)
    print("Benchmark metrics:", report_json)
    print("Benchmark summary:", report_csv)
    print("Benchmark report:", report_md)


if __name__ == "__main__":
    main()
