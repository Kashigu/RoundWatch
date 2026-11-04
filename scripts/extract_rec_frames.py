import cv2
import json

cap = cv2.VideoCapture('data/recordings/carla_sumo_roundabout.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames in recording:", total)

check_frames = [0, 80, 150, 250, 350, 450, 550, 650, 750]
for idx in check_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'rec_frame_{idx}.png', frame)
        print(f"Saved rec_frame_{idx}.png")
cap.release()

