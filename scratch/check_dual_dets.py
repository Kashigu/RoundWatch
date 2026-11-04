import cv2
import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.perception.detector import VehicleDetector

detector = VehicleDetector(model_name="yolov8s.pt", confidence_thresh=0.25, device="cuda:0")
cap = cv2.VideoCapture('data/recordings/carla_sumo_roundabout.mp4')
pane_w = 1280

# Check frames 170, 200, 230, 410
for fid in [170, 200, 230, 410]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    dets_s = detector.detect(frame[:, :pane_w])
    dets_n = detector.detect(frame[:, pane_w:])
    print(f"\n--- Frame {fid} ---")
    print(f"South ({len(dets_s)}):", [(d['class_name'], d['bbox']) for d in dets_s])
    print(f"North ({len(dets_n)}):", [(d['class_name'], d['bbox']) for d in dets_n])

