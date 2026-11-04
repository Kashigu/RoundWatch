import json
import math

tracks = json.load(open('output/tracks.json', 'r', encoding='utf-8'))

print(f"Total tracks analyzed: {len(tracks)}")
suspicious_tracks = []

for tid, t in tracks.items():
    bboxes = t['history_bboxes']
    if len(bboxes) < 2:
        continue
    max_jump = 0.0
    jump_f = 0
    for i in range(1, len(bboxes)):
        b0, b1 = bboxes[i - 1], bboxes[i]
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        dist = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        if dist > max_jump:
            max_jump = dist
            jump_f = t['first_frame'] + i
    widths = [b[2] - b[0] for b in bboxes]
    heights = [b[3] - b[1] for b in bboxes]

    # Check cross-boundary jump (from left pane <1280 to right pane >1280)
    has_cross_pane = False
    for i in range(1, len(bboxes)):
        if (bboxes[i - 1][0] < 1280 and bboxes[i][0] >= 1280) or (bboxes[i - 1][0] >= 1280 and bboxes[i][0] < 1280):
            has_cross_pane = True
            break

    if max_jump > 60 or max(widths) > 350 or max(heights) > 350 or has_cross_pane:
        suspicious_tracks.append((tid, max_jump, jump_f, max(widths), max(heights), has_cross_pane))
        print(f"Track #{tid} [{t.get('class_name')}]: max_jump={max_jump:.1f}px (frame {jump_f}), max_w={max(widths)}, max_h={max(heights)}, cross_pane={has_cross_pane}, frames={t['first_frame']}..{t['last_frame']}")

print(f"Found {len(suspicious_tracks)} suspicious/flying tracks out of {len(tracks)}")

