import cv2
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.perception.detector import VehicleDetector

detector = VehicleDetector(model_name="yolov8s.pt", confidence_thresh=0.25, device="cuda:0")
cap = cv2.VideoCapture('data/recordings/carla_sumo_roundabout.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pane_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2

print(f"Total video frames: {total}, pane_w: {pane_w}")

for fid in range(50, total, 60):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if not ret: break
    dets_s = detector.detect(frame[:, :pane_w])
    dets_n = detector.detect(frame[:, pane_w:])
    print(f"Frame {fid:3d}: South={len(dets_s)} boxes, North={len(dets_n)} boxes")
    for d in dets_s:
        bx = d['bbox']
        print(f"   [South] {d['class_name']} conf={d['conf']:.2f} box={bx} w={bx[2]-bx[0]} h={bx[3]-bx[1]}")
    for d in dets_n:
        bx = d['bbox']
        print(f"   [North] {d['class_name']} conf={d['conf']:.2f} box={bx} w={bx[2]-bx[0]} h={bx[3]-bx[1]}")

