"""Export recorded CARLA datasets (scripts/record_tracking_dataset.py) to YOLO detection format.

Labels are the visible vehicle boxes from CARLA instance segmentation, using COCO class indices
(2 car, 3 motorcycle, 5 bus, 7 truck), so a fine-tuned model is a drop-in replacement for the
stock COCO weights in VehicleDetector.

Usage:
    .venv310\\Scripts\\python.exe scripts/export_yolo_dataset.py --train data/eval/train_a data/eval/train_b \\
        --val data/eval/val --out data/yolo_carla --stride 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]


def coco_class(type_id: str) -> int:
    t = type_id.lower()
    if any(k in t for k in ("carlacola", "cybertruck", "firetruck", "ambulance")):
        return 7
    if any(k in t for k in ("yamaha", "harley", "kawasaki", "vespa")):
        return 3
    if any(k in t for k in ("gazelle", "diamondback", "crossbike")):
        return 1
    if "fusorosa" in t:
        return 5
    return 2


def export(src_dirs, split: str, out: Path, stride: int, min_px: int, min_side: int) -> int:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for src in map(Path, src_dirs):
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        W, H = meta["width"], meta["height"]
        tag = src.name
        for i, line in enumerate((src / "gt.jsonl").read_text(encoding="utf-8").splitlines()):
            if i % stride:
                continue
            fr = json.loads(line)
            for cam in ("south", "north"):
                rows = []
                for v in fr["vehicles"]:
                    o = v.get(cam)
                    if not o or not o["vis_box"] or o["vis_px"] < min_px:
                        continue
                    x1, y1, x2, y2 = o["vis_box"]
                    if x2 - x1 < min_side or y2 - y1 < min_side:
                        continue
                    rows.append(f"{coco_class(v['type'])} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                name = f"{tag}_{cam}_{fr['frame']:06d}"
                img = src / cam / f"{fr['frame']:06d}.jpg"
                dst = out / "images" / split / f"{name}.jpg"
                try:
                    os.link(img, dst)  # hard link: no extra disk space
                except OSError:
                    shutil.copy2(img, dst)
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows), encoding="utf-8")
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="data/yolo_carla")
    ap.add_argument("--stride", type=int, default=4, help="use every Nth frame (consecutive frames are near-duplicates)")
    ap.add_argument("--min-px", type=int, default=60, help="min visible pixels for a label")
    ap.add_argument("--min-side", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    nt = export(args.train, "train", out, args.stride, args.min_px, args.min_side)
    nv = export(args.val, "val", out, args.stride, args.min_px, args.min_side)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(COCO_NAMES)), encoding="utf-8")
    print(f"[export] train images: {nt} | val images: {nv} -> {out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
