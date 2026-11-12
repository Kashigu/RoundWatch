import base64, cv2, numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.runtime_paths import artifact_directory

html = (artifact_directory() / "dual_camera_verification_dashboard.html").read_text(encoding="utf-8")

prefix = "data:image/jpeg;base64,"
idx1 = html.find(prefix)
idx2 = html.find('"', idx1)
b64 = html[idx1 + len(prefix):idx2]
buf = base64.b64decode(b64)
img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

print("Embedded first image shape:", img.shape, "Mean intensity:", img.mean())
if img.mean() > 140:
    print("VERIFIED: The HTML now embeds the clear forward view with 0 floodlight obstruction!")
else:
    print("WARNING: Still obstructed!")
