"""Validate YOLO models on exported benchmark datasets and tabulate the results.

Runs every combination of dataset x model x image size and writes
``summary.json`` (all metrics, per class) and ``summary.md`` (tables) to
``runs/benchmark/<name>/``. By default it compares YOLOv8 and YOLO11
(nano and small) at 1920 px; missing weights are downloaded to the project root.

Examples:
    # every val_*fps dataset of one recording, default models and image size
    python scripts/run_yolo_benchmark.py data/yolo_benchmark/clear_noon_02

    # specific datasets, chosen models and several image sizes
    python scripts/run_yolo_benchmark.py data/yolo_benchmark/clear_noon_02/val_20fps \\
        --models yolov8s yolo11s --imgsz 640 1280 1920
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_paths import PROJECT_ROOT

# COCO-pretrained detectors compared by default: YOLOv8 and its successor YOLO11.
DEFAULT_MODELS = ["yolov8n", "yolov8s", "yolo11n", "yolo11s"]


def find_datasets(paths: list[Path]) -> list[Path]:
    """Resolve dataset folders (containing dataset.yaml) from files or parent folders."""
    found: list[Path] = []
    for path in paths:
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            found.append(path.parent)
        elif (path / "dataset.yaml").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p.parent for p in path.glob("*/dataset.yaml")))
        else:
            raise SystemExit(f"No dataset found at {path}")
    if not found:
        raise SystemExit("No datasets found (expected folders containing dataset.yaml)")

    def sort_key(p: Path) -> tuple[str, float]:
        name = p.name
        if name.startswith("val_") and name.endswith("fps"):
            try:
                return (str(p.parent), float(name[4:-3]))
            except ValueError:
                pass
        return (str(p), 0.0)

    return sorted(dict.fromkeys(found), key=sort_key)


def default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def validate(model_name: str, dataset: Path, imgsz: int, device: str, out_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    weights = Path(model_name if model_name.endswith(".pt") else f"{model_name}.pt")
    if not weights.is_absolute() and weights.parent == Path("."):
        weights = PROJECT_ROOT / weights  # keep (and download) weights in the project root
    model = YOLO(str(weights))
    started = time.perf_counter()
    metrics = model.val(
        data=str(dataset / "dataset.yaml"),
        imgsz=imgsz,
        batch=8 if device != "cpu" else 1,
        device=device,
        project=str(out_dir),
        name=f"{dataset.name}_{Path(model_name).stem}_{imgsz}",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    per_class = {}
    for i, class_index in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class[model.names[int(class_index)]] = {
            "P": round(float(p), 3), "R": round(float(r), 3),
            "mAP50": round(float(ap50), 3), "mAP50_95": round(float(ap), 3),
        }
    return {
        "dataset": dataset.name,
        "dataset_path": str(dataset),
        "images": len(list((dataset / "images").glob("*.jpg"))),
        "model": Path(model_name).stem,
        "imgsz": imgsz,
        "device": device,
        "P": round(float(metrics.box.mp), 3),
        "R": round(float(metrics.box.mr), 3),
        "mAP50": round(float(metrics.box.map50), 3),
        "mAP50_95": round(float(metrics.box.map), 3),
        "inference_ms": round(metrics.speed["inference"], 2),
        "total_ms": round(sum(metrics.speed.values()), 2),
        "wall_s": round(time.perf_counter() - started, 1),
        "per_class": per_class,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    classes = sorted({c for row in rows for c in row["per_class"]})
    head = ["Dataset", "Images", "Model", "imgsz", "P", "R", "mAP50", "mAP50-95"]
    head += [f"{c} mAP50" for c in classes] + ["ms/img (inference)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [row["dataset"], row["images"], row["model"], row["imgsz"], row["P"], row["R"],
                 row["mAP50"], row["mAP50_95"]]
        cells += [row["per_class"].get(c, {}).get("mAP50", "—") for c in classes]
        cells.append(row["inference_ms"])
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    # Model comparison: one row per dataset/image size, one column per model.
    models = list(dict.fromkeys(row["model"] for row in rows))
    if len(models) > 1:
        lines += ["", "**Model comparison (mAP50 / mAP50-95, inference ms/img)**", ""]
        head = ["Dataset", "imgsz"] + models
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        by_key = {(r["dataset"], r["imgsz"], r["model"]): r for r in rows}
        for dataset, imgsz in dict.fromkeys((r["dataset"], r["imgsz"]) for r in rows):
            cells = [dataset, str(imgsz)]
            for model in models:
                r = by_key.get((dataset, imgsz, model))
                cells.append(
                    f"{r['mAP50']:.3f} / {r['mAP50_95']:.3f}, {r['inference_ms']:.0f} ms" if r else "—"
                )
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", type=Path, nargs="+", help="Dataset folder(s), dataset.yaml, or a parent folder")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Ultralytics model names or .pt paths (default: {' '.join(DEFAULT_MODELS)}); "
        "missing official weights are downloaded automatically",
    )
    parser.add_argument("--imgsz", type=int, nargs="+", default=[1920])
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu' (default: GPU if available)")
    parser.add_argument("--name", default=None, help="Output folder name under runs/benchmark (default: dataset folder name)")
    args = parser.parse_args()

    datasets = find_datasets(args.datasets)
    device = args.device or default_device()
    parents = {d.parent for d in datasets}
    if args.name:
        name = args.name
    elif len(datasets) == 1:
        name = f"{datasets[0].parent.name}_{datasets[0].name}"
    elif len(parents) == 1:
        name = datasets[0].parent.name
    else:
        name = "_".join(sorted(p.name for p in parents))
    out_dir = PROJECT_ROOT / "runs" / "benchmark" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(datasets) * len(args.models) * len(args.imgsz)
    print(f"Running {total} validations on device {device}; results in {out_dir}")
    rows = []
    for dataset in datasets:
        for model_name in args.models:
            for imgsz in args.imgsz:
                row = validate(model_name, dataset, imgsz, device, out_dir)
                rows.append(row)
                print(f"  {row['dataset']:>12} {row['model']:>8} @ {imgsz}: "
                      f"mAP50 {row['mAP50']:.3f}  mAP50-95 {row['mAP50_95']:.3f}  "
                      f"R {row['R']:.3f}  {row['inference_ms']:.1f} ms/img")
                (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    (out_dir / "summary.md").write_text(markdown(rows), encoding="utf-8")
    print("\n" + markdown(rows))
    print(f"Saved {out_dir / 'summary.json'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
