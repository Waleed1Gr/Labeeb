import os
import cv2
import numpy as np
from ultralytics import YOLO

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8s.pt")
FACE_YOLO_WEIGHTS = os.getenv("FACE_YOLO_WEIGHTS", "face_yolov8s.pt")

# Load person+phone model
try:
    model = YOLO(YOLO_WEIGHTS)
except Exception as e:
    print(f"❌ Failed to load YOLO model ({YOLO_WEIGHTS}): {e}")
    model = None

# Load face recognition model
try:
    face_model = YOLO(FACE_YOLO_WEIGHTS)
except Exception as e:
    print(f"❌ Failed to load Face YOLO model ({FACE_YOLO_WEIGHTS}): {e}")
    face_model = None

PERSON_LABELS = {"person"}
PHONE_LABELS = {"cell phone", "mobile phone", "phone", "cellphone"}

def detect_objects(frame: np.ndarray) -> dict:
    """
    Run both YOLO models on the frame, return:
    {
        "people": [(x1, y1, x2, y2), ...],
        "phones": [(x1, y1, x2, y2), ...],
        "names": [name1, name2, ...]
    }
    """
    if model is None or face_model is None:
        return {"people": [], "phones": [], "names": []}

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Detect people & phones ---
        results = model.predict(source=rgb_frame, verbose=False)[0]
        cls_indices = results.boxes.cls.cpu().numpy().astype(int)
        xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        names = model.names

        people_boxes, phone_boxes = [], []

        for idx, cls_id in enumerate(cls_indices):
            cls_name = names.get(cls_id, "").lower()
            x1, y1, x2, y2 = xyxy[idx].tolist()

            if cls_name in PERSON_LABELS:
                people_boxes.append((x1, y1, x2, y2))
            elif cls_name in PHONE_LABELS:
                phone_boxes.append((x1, y1, x2, y2))

        # --- Detect face names only ---
        face_results = face_model.predict(source=rgb_frame, verbose=False)[0]
        face_cls = face_results.boxes.cls.cpu().numpy().astype(int)
        face_names = face_model.names

        detected_names = []
        for cls_id in face_cls:
            name = face_names.get(cls_id, f"Unknown_{cls_id}")
            if name not in detected_names:
                detected_names.append(name)

        return {
            "people": people_boxes,
            "phones": phone_boxes,
            "names": detected_names
        }

    except Exception as e:
        print(f"YOLO detect error: {e}")
        return {"people": [], "phones": [], "names": []}
