import cv2
import json

tracks = json.load(open('output/tracks.json', 'r', encoding='utf-8'))
cap = cv2.VideoCapture('output/roundwatch_ar_annotated.mp4')

for fid in [180, 420, 540]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        active = [tid for tid, d in tracks.items() if d['first_frame'] <= fid <= d['last_frame']]
        print(f"=== Frame {fid} (Active vehicles: {len(active)}) ===")
        for tid in active[:4]:
            d = tracks[tid]
            offset = fid - d['first_frame']
            bbox = d['history_bboxes'][offset]
            print(f"  Vehicle #{tid} [{d.get('class_name')}]: conf={d.get('conf'):.2f}, bbox={bbox}")
cap.release()

