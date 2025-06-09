# api_clients/vision_api.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Load YOLO Model Locally
#
# If you trained a custom .pt, set YOLO_WEIGHTS in .env,
# otherwise it defaults to “yolov8s.pt” (Ultralytics’ official small model).
# -----------------------------------------------------------------------------

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")

try:
    model = YOLO(YOLO_WEIGHTS)
except Exception as e:
    print(f"❌ Failed to load YOLO model ({YOLO_WEIGHTS}): {e}")
    model = None

# We only care about “person” and “cell phone” labels
PERSON_LABELS = {"person"}
PHONE_LABELS = {"cell phone", "mobile phone", "phone", "cellphone"}

def detect_objects(frame: np.ndarray) -> dict:
    """
    Run local YOLO on the BGR frame, return {"people": [...], "phones": [...]},
    where each box is (x1, y1, x2, y2).
    """
    if model is None:
        return {"people": [], "phones": []}

    try:
        # YOLO will handle resizing internally; just convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=rgb_frame, verbose=False)
        r = results[0]

        cls_indices = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)

        people_boxes = []
        phone_boxes = []
        names = model.names

        for idx, cls_id in enumerate(cls_indices):
            cls_name = names.get(cls_id, "").lower()
            x1, y1, x2, y2 = xyxy[idx].tolist()
            if cls_name in PERSON_LABELS:
                people_boxes.append((x1, y1, x2, y2))
            elif cls_name in PHONE_LABELS:
                phone_boxes.append((x1, y1, x2, y2))

        return {"people": people_boxes, "phones": phone_boxes}

    except Exception as e:
        print(f"YOLO detect error: {e}")
        return {"people": [], "phones": []}
