"""Export the recorded ground-truth clips in data/eval/ as YOLO validation sets, per condition.

Each clip in ``data/eval/<clip>/`` (frames in ``<camera>/NNNNNN.jpg``, labels in ``gt.jsonl``
with instance-segmentation ``vis_box`` per camera, weather in ``meta.json``) becomes one
dataset. ``--shake`` adds copies with simulated wind sway of the pole camera (``PoleSway``
applied as a rotation homography plus motion blur; labels warped with the same homography).

Output: ``data/yolo_benchmark/<name>/<clip>_<weather>[_shake<deg>]/{images,labels,dataset.yaml}``,
ready for ``scripts/run_yolo_benchmark.py data/yolo_benchmark/<name>``.

Example:
    .venv310\\Scripts\\python.exe scripts/export_conditions_benchmark.py --fps 2 --shake 0.5 1.0 --shake-clips train_s101_ClearNoon
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_yolo_benchmark import COCO_NAMES, class_id_for_vehicle, yolo_annotation
from scripts.runtime_paths import PROJECT_ROOT
from src.simulation.conditions import PoleSway, shake_frame


def load_gt(clip: Path, camera: str, min_px: int) -> dict[int, list]:
    boxes: dict[int, list] = {}
    with open(clip / "gt.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows = []
            for v in rec["vehicles"]:
                cam = v.get(camera)
                if not cam or not cam.get("vis_box") or cam.get("vis_px", 0) < min_px:
                    continue
                rows.append((class_id_for_vehicle(v.get("type", "")), tuple(cam["vis_box"])))
            boxes[int(rec["frame"])] = rows
    return boxes


def export_clip(clip: Path, out_dir: Path, camera: str, fps: float, shake: float, seed: int,
                min_px: int) -> tuple[int, int]:
    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    src_fps = float(meta.get("fps", 20.0))
    fov = float(meta.get("fov", 90.0))
    step = max(1, round(src_fps / fps))
    gt = load_gt(clip, camera, min_px)
    sway = PoleSway(amplitude_deg=shake, pole_height=meta["cameras"][camera]["position"][2], seed=seed) if shake > 0 else None

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n_img = n_lab = 0
    for frame in sorted(gt)[::step]:
        src = clip / camera / f"{frame:06d}.jpg"
        if not src.is_file():
            continue
        stem = f"frame_{frame:06d}"
        boxes = gt[frame]
        if sway is None:
            shutil.copyfile(src, out_dir / "images" / f"{stem}.jpg")
            w, h = int(meta["width"]), int(meta["height"])
        else:
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            img, boxes = shake_frame(img, boxes, sway, frame / src_fps, fov_deg=fov)
            cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows = [yolo_annotation(c, b, w, h) for c, b in boxes]
        (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        n_img += 1
        n_lab += len(rows)

    (out_dir / "dataset.yaml").write_text(
        "train: null\nval: images\nnames:\n" + "".join(f"  {i}: {n!r}\n" for i, n in enumerate(COCO_NAMES)),
        encoding="utf-8",
    )
    info = {"clip": clip.name, "weather": meta.get("weather", "unknown"), "camera": camera, "fps": fps,
            "camera_shake_deg": shake, "shake_seed": seed, "images": n_img, "labels": n_lab}
    (out_dir / "conditions.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return n_img, n_lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", type=Path, default=PROJECT_ROOT / "data" / "eval")
    ap.add_argument("--clips", nargs="*", default=None, help="Clip folder names (default: all with meta.json + gt.jsonl)")
    ap.add_argument("--name", default="conditions", help="Output folder under data/yolo_benchmark")
    ap.add_argument("--camera", choices=("south", "north"), default="south")
    ap.add_argument("--fps", type=float, default=2.0, help="Sampling rate (default 2)")
    ap.add_argument("--shake", type=float, nargs="*", default=[], help="Extra shaken copies, e.g. --shake 0.5 1.0")
    ap.add_argument("--shake-clips", nargs="*", default=None, help="Clips to shake (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-visible-px", type=int, default=25)
    args = ap.parse_args()

    clips = [args.eval_root / c for c in args.clips] if args.clips else sorted(
        p for p in args.eval_root.iterdir() if (p / "meta.json").is_file() and (p / "gt.jsonl").is_file())
    root = PROJECT_ROOT / "data" / "yolo_benchmark" / args.name
    for clip in clips:
        weather = json.loads((clip / "meta.json").read_text(encoding="utf-8")).get("weather", "unknown")
        base = clip.name if weather in clip.name else f"{clip.name}_{weather}"
        variants = [0.0] + [s for s in args.shake if args.shake_clips is None or clip.name in args.shake_clips]
        for shake in variants:
            out = root / (base + (f"_shake{shake:g}" if shake else ""))
            if out.exists():
                shutil.rmtree(out)
            n_img, n_lab = export_clip(clip, out, args.camera, args.fps, shake, args.seed, args.min_visible_px)
            print(f"{out.name:40s} {n_img:4d} images {n_lab:5d} labels")
    print(f"\nDatasets: {root}")


if __name__ == "__main__":
    main()
