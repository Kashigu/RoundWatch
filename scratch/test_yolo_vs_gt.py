import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import json
from src.perception.detector import VehicleDetector

detector = VehicleDetector(model_name="yolov8s.pt", confidence_thresh=0.25, device="cuda:0")
cap = cv2.VideoCapture('data/recordings/carla_sumo_roundabout.mp4')
gt = json.load(open('data/recordings/carla_sumo_gt.json'))
gt_by_frame = {}
for r in gt:
    f = r['frame_idx']
    if f not in gt_by_frame: gt_by_frame[f] = []
    gt_by_frame[f].append(r)

cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
ret, frame = cap.read()
pane_w = frame.shape[1] // 2
frame_s = frame[:, :pane_w]
frame_n = frame[:, pane_w:]

dets_s = detector.detect(frame_s)
dets_n = detector.detect(frame_n)

print(f"=== Frame 180 ===")
print("Ground Truth in South:", [(r['sumo_id'], r['bbox_south']) for r in gt_by_frame.get(180, []) if r.get('bbox_south')])
print("YOLO in South:", [(d['class_name'], d['conf'], d['bbox']) for d in dets_s])
print("Ground Truth in North:", [(r['sumo_id'], r['bbox_north']) for r in gt_by_frame.get(180, []) if r.get('bbox_north')])
print("YOLO in North:", [(d['class_name'], d['conf'], d['bbox']) for d in dets_n])
