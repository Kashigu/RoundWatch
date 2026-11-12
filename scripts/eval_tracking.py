"""Evaluate the perception + tracking stack on a recorded ground-truth dataset.

Usage:
    .venv310\\Scripts\\python.exe scripts/eval_tracking.py --data data/eval/roundabout_2lane
    .venv310\\Scripts\\python.exe scripts/eval_tracking.py --model yolov8m.pt --res 1920x1080 --imgsz 1920

Writes the full report as JSON (--out) and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from src.evaluation.tracking_eval import EvalConfig, evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval/roundabout_2lane")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--res", default="1280x720", help="resize frames before inference, e.g. 1280x720 or 1920x1080")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--tracker-cfg", default="configs/botsort.yaml")
    ap.add_argument("--max-age", type=int, default=40)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ground-point", default="box75", choices=["box75", "bottom"])
    ap.add_argument("--no-merge", action="store_true", help="disable duplicate-vehicle merging in the fusion")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = tuple(int(v) for v in args.res.lower().split("x")) if args.res else None
    cfg = EvalConfig(model=args.model, res=res, imgsz=args.imgsz, tracker_cfg=args.tracker_cfg,
                     fusion={"max_age_frames": args.max_age, "min_hits": args.min_hits, "ground_point": args.ground_point,
                             "merge_duplicates": not args.no_merge}, max_frames=args.max_frames)
    r = evaluate(args.data, cfg)
    out = Path(args.out or f"output/eval/{Path(args.model).stem}_{args.res}_{args.imgsz or 'auto'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")

    g = r["global"]
    print(f"\n== {args.model} | frames {args.res} | imgsz {args.imgsz or 'auto'} | {r['fps_pipeline']} FPS (2 cams)")
    for c, s in r["per_camera"].items():
        print(f"  {c:5s}: recall {s['recall']:.3f}  precision {s['precision']:.3f}  local ID switches {s['id_switches']}")
    print(f"  global: MOTA {g['MOTA']:.3f}  coverage {g['coverage']:.3f}  ID switches {g['id_switches']} {g['id_switch_kinds']}  "
          f"false tracks {g['false_tracks']}")
    print(f"          lost-in-both episodes (>=3 frames): {g['lost_episodes_3plus_frames']}  frames by cause: {g['lost_frames_by_cause']}")
    print(f"  report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
