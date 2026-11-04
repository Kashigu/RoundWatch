import cv2
import json
import math
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.perception.detector import VehicleDetector

def pixel_to_road_ray(u, v, cam_pos, cam_rot, fov=90.0, w=1280, h=720, ground_z=0.0):
    focal = (w / 2.0) / math.tan(math.radians(fov / 2.0))
    x_cam = (u - (w / 2.0)) / focal
    y_cam = (v - (h / 2.0)) / focal
    z_cam = 1.0

    v_c = np.array([z_cam, x_cam, -y_cam])
    v_c /= np.linalg.norm(v_c)

    pitch = math.radians(cam_rot[0])
    yaw = math.radians(cam_rot[1])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_w2c = R_pitch @ R_yaw
    R_c2w = R_w2c.T

    ray_w = R_c2w @ v_c
    if abs(ray_w[2]) < 1e-4 or ray_w[2] >= 0:
        return None

    t = (ground_z - cam_pos[2]) / ray_w[2]
    if t <= 0:
        return None

    return float(cam_pos[0] + t * ray_w[0]), float(cam_pos[1] + t * ray_w[1])


class DynamicMultiCameraTracker:
    """Pure dynamic multi-camera tracking engine without GT or pre-defined IDs."""

    def __init__(self, cam_pos_s, cam_rot_s, cam_pos_n, cam_rot_n, max_age_frames=45, min_hits=3):
        self.cam_pos_s = cam_pos_s
        self.cam_rot_s = cam_rot_s
        self.cam_pos_n = cam_pos_n
        self.cam_rot_n = cam_rot_n
        self.max_age = max_age_frames
        self.min_hits = min_hits
        self.next_id = 1

        self.tentative_tracks = []
        self.confirmed_tracks = []
        self.completed_tracks = {}

    def update(self, dets_s, dets_n, frame_idx, dt=0.05, pane_w=1280, height=720):
        # 1. Project detections to road coordinates
        obs_s = []
        for d in dets_s:
            bx = d['bbox']
            fu, fv = (bx[0] + bx[2]) / 2.0, float(bx[3])
            pt = pixel_to_road_ray(fu, fv, self.cam_pos_s, self.cam_rot_s, fov=90.0, w=pane_w, h=height)
            if pt:
                obs_s.append({'pane': 'south', 'bbox': bx, 'road': pt, 'class_name': d['class_name'], 'conf': d['conf']})

        obs_n = []
        for d in dets_n:
            bx = d['bbox']
            fu, fv = (bx[0] + bx[2]) / 2.0, float(bx[3])
            pt = pixel_to_road_ray(fu, fv, self.cam_pos_n, self.cam_rot_n, fov=90.0, w=pane_w, h=height)
            if pt:
                obs_n.append({'pane': 'north', 'bbox': bx, 'road': pt, 'class_name': d['class_name'], 'conf': d['conf']})

        # 2. Fuse simultaneous detections (same car seen in South and North at same moment)
        fused_obs = []
        used_s = set()
        used_n = set()

        for is_idx, os_item in enumerate(obs_s):
            best_in = None
            best_dist = float('inf')
            for in_idx, on_item in enumerate(obs_n):
                if in_idx in used_n: continue
                d = math.hypot(os_item['road'][0] - on_item['road'][0], os_item['road'][1] - on_item['road'][1])
                if d < best_dist and d < 6.5:
                    best_dist = d
                    best_in = in_idx

            if best_in is not None:
                used_s.add(is_idx)
                used_n.add(best_in)
                on_item = obs_n[best_in]
                avg_road = (
                    0.5 * (os_item['road'][0] + on_item['road'][0]),
                    0.5 * (os_item['road'][1] + on_item['road'][1]),
                )
                fused_obs.append({
                    'bbox_s': os_item['bbox'],
                    'bbox_n': on_item['bbox'],
                    'road': avg_road,
                    'class_name': os_item['class_name'] if os_item['conf'] >= on_item['conf'] else on_item['class_name'],
                    'conf': max(os_item['conf'], on_item['conf']),
                })
            else:
                fused_obs.append({
                    'bbox_s': os_item['bbox'],
                    'bbox_n': None,
                    'road': os_item['road'],
                    'class_name': os_item['class_name'],
                    'conf': os_item['conf'],
                })

        for in_idx, on_item in enumerate(obs_n):
            if in_idx not in used_n:
                fused_obs.append({
                    'bbox_s': None,
                    'bbox_n': on_item['bbox'],
                    'road': on_item['road'],
                    'class_name': on_item['class_name'],
                    'conf': on_item['conf'],
                })

        # 3. Associate with confirmed tracks first
        unmatched_obs = list(range(len(fused_obs)))
        for trk in self.confirmed_tracks:
            pred_x = trk['road'][0] + trk['vel'][0] * dt
            pred_y = trk['road'][1] + trk['vel'][1] * dt

            best_oi = None
            best_dist = float('inf')
            for oi in unmatched_obs:
                fo = fused_obs[oi]
                d = math.hypot(fo['road'][0] - pred_x, fo['road'][1] - pred_y)
                # Motion gating: up to 8m (or 14m if coasting across fountain)
                max_gate = 8.0 if (frame_idx - trk['last_seen'] <= 5) else 14.0
                if d < best_dist and d < max_gate:
                    best_dist = d
                    best_oi = oi

            if best_oi is not None:
                fo = fused_obs[best_oi]
                unmatched_obs.remove(best_oi)
                vx = (fo['road'][0] - trk['road'][0]) / max(dt, (frame_idx - trk['last_seen']) * dt)
                vy = (fo['road'][1] - trk['road'][1]) / max(dt, (frame_idx - trk['last_seen']) * dt)
                trk['vel'] = (0.7 * trk['vel'][0] + 0.3 * vx, 0.7 * trk['vel'][1] + 0.3 * vy)
                trk['road'] = fo['road']
                trk['last_seen'] = frame_idx
                trk['hits'] += 1
                trk['active_s'] = fo['bbox_s']
                trk['active_n'] = fo['bbox_n']
                trk['conf'] = fo['conf']
            else:
                trk['active_s'] = None
                trk['active_n'] = None

        # 4. Associate remaining observations with tentative tracks
        remaining_unmatched = []
        for oi in unmatched_obs:
            fo = fused_obs[oi]
            matched_ten = False
            for ten in self.tentative_tracks:
                d = math.hypot(fo['road'][0] - ten['road'][0], fo['road'][1] - ten['road'][1])
                if d < 7.0:
                    ten['road'] = fo['road']
                    ten['last_seen'] = frame_idx
                    ten['hits'] += 1
                    ten['active_s'] = fo['bbox_s']
                    ten['active_n'] = fo['bbox_n']
                    matched_ten = True
                    # Promote to confirmed if reached min_hits!
                    if ten['hits'] >= self.min_hits:
                        ten['id'] = self.next_id
                        print(f"[Dynamic Tracker] Frame {frame_idx:3d}: CONFIRMED VEHICLE -> Assigned Dynamic ID #{self.next_id} ({ten['class_name']})")
                        self.next_id += 1
                        self.confirmed_tracks.append(ten)
                        self.tentative_tracks.remove(ten)
                    break
            if not matched_ten:
                remaining_unmatched.append(oi)

        # 5. Create new tentative track (MUST prove itself over min_hits before getting an ID)
        for oi in remaining_unmatched:
            fo = fused_obs[oi]
            r_center = math.hypot(fo['road'][0] - 2.0, fo['road'][1] - 6.0)
            if r_center < 38.0:
                self.tentative_tracks.append({
                    'class_name': fo['class_name'],
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'hits': 1,
                    'road': fo['road'],
                    'vel': (0.0, 0.0),
                    'active_s': fo['bbox_s'],
                    'active_n': fo['bbox_n'],
                    'conf': fo['conf'],
                })

        # 6. Prune stale tentative tracks (> 5 frames unseen)
        self.tentative_tracks = [t for t in self.tentative_tracks if frame_idx - t['last_seen'] <= 5]

        # 7. Archive expired confirmed tracks (> max_age frames unseen)
        surviving_conf = []
        for trk in self.confirmed_tracks:
            if frame_idx - trk['last_seen'] > self.max_age:
                self.completed_tracks[str(trk['id'])] = trk
                print(f"[Dynamic Tracker] Frame {frame_idx:3d}: Vehicle #{trk['id']} EXITED roundabout (lifespan: {trk['first_seen']}..{trk['last_seen']}, hits: {trk['hits']})")
            else:
                surviving_conf.append(trk)
        self.confirmed_tracks = surviving_conf

        # Return confirmed active tracks visible in current frame
        return [t for t in self.confirmed_tracks if (t['active_s'] is not None or t['active_n'] is not None)]

def test_pure_dynamic_run():
    detector = VehicleDetector(model_name="yolov8s.pt", confidence_thresh=0.25, device="cuda:0")
    cam_pos_s = [0.71, -26.2, 11.5]
    cam_rot_s = [-28.0, 88.0, 0.0]
    cam_pos_n = [-2.31, 27.2, 11.5]
    cam_rot_n = [-28.0, -80.0, 0.0]

    tracker = DynamicMultiCameraTracker(cam_pos_s, cam_rot_s, cam_pos_n, cam_rot_n, max_age_frames=45, min_hits=3)

    cap = cv2.VideoCapture("data/recordings/carla_sumo_roundabout.mp4")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pane_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Running pure dynamic tracking on {total_frames} frames...")
    t0 = time.time()

    for fidx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        frame_s = frame[:, :pane_w]
        frame_n = frame[:, pane_w:]

        dets_s = detector.detect(frame_s)
        dets_n = detector.detect(frame_n)

        active = tracker.update(dets_s, dets_n, fidx, dt=0.05, pane_w=pane_w, height=height)

        if fidx % 150 == 0:
            print(f"  Frame {fidx:3d}/{total_frames} | Active: {len(active)} vehicles (Confirmed IDs so far: {tracker.next_id - 1})")

    cap.release()
    print(f"\nCompleted in {time.time()-t0:.2f}s!")
    all_tracks = {**tracker.completed_tracks, **{str(t['id']): t for t in tracker.confirmed_tracks}}
    print(f"Total dynamic vehicles confirmed: {len(all_tracks)}")
    for tid, t in sorted(all_tracks.items(), key=lambda x: int(x[0])):
        print(f"  Vehicle #{tid} [{t['class_name']}]: frames {t['first_seen']}..{t['last_seen']} (hits: {t['hits']})")

if __name__ == "__main__":
    test_pure_dynamic_run()
