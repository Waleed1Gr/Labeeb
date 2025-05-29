import cv2
import time
import threading
from queue import Queue
from audio.speak import speak_warning
from ultralytics import YOLO


# ============== CAMERA PERSON & PHONE DETECTOR ==============
def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def phone_person_detector():
    print("📸 بدأ كشف الكاميرا...")
    # Use smaller model but keep same detection capabilities
    model = YOLO('models/yolov5n.pt')
    model.conf = 0.4

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ الكاميرا غير متوفرة")
        return

    # Preserve all tracking variables
    unique_people = {}
    person_id_counter = 0
    phone_start_times = {}
    phone_warning_given = {}
    phone_put_down_time = {}

    def current_time():
        return time.time()

    # Frame processing queue
    frame_queue = Queue(maxsize=2)
    running = threading.Event()
    running.set()

    def capture_frames():
        while running.is_set():
            ret, frame = cap.read()
            if ret:
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame)
            time.sleep(0.01)

    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.start()

    skip_frames = 0

    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        skip_frames += 1

        if skip_frames % 2 != 0:
            continue

        # Scale down for processing but keep original for display
        frame_small = cv2.resize(frame, (416, 416))
        results = model(frame_small)

        # Scale detections back to original frame size
        scale_x = frame.shape[1] / 416
        scale_y = frame.shape[0] / 416

        detected_people = []
        detected_phones = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1.item() * scale_x),
                        int(y1.item() * scale_y),
                        int(x2.item() * scale_x),
                        int(y2.item() * scale_y),
                    )
                    detected_people.append((x1, y1, x2, y2))
                elif cls == 67:  # Phone
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1.item() * scale_x),
                        int(y1.item() * scale_y),
                        int(x2.item() * scale_x),
                        int(y2.item() * scale_y),
                    )
                    detected_phones.append((x1, y1, x2, y2))

        for px1, py1, px2, py2 in detected_people:
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            min_dist = float("inf")
            closest_phone = None

            for phx1, phy1, phx2, phy2 in detected_phones:
                phone_center = ((phx1 + phx2) // 2, (phy1 + phy2) // 2)
                dist = (
                    (person_center[0] - phone_center[0]) ** 2
                    + (person_center[1] - phone_center[1]) ** 2
                ) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_phone = (phx1, phy1, phx2, phy2)

            found_person_id = None
            for pid, box in unique_people.items():
                if intersection_over_union((px1, py1, px2, py2), box) > 0.3:
                    found_person_id = pid
                    unique_people[pid] = (px1, py1, px2, py2)
                    break
            if found_person_id is None:
                person_id_counter += 1
                found_person_id = person_id_counter
                unique_people[found_person_id] = (px1, py1, px2, py2)

            if closest_phone and min_dist < 150:
                now = current_time()
                if found_person_id not in phone_start_times:
                    phone_start_times[found_person_id] = now
                    phone_warning_given[found_person_id] = False

                duration = now - phone_start_times[found_person_id]
                if duration > 10 and not phone_warning_given[found_person_id]:
                    speak_warning("يالحبيب، لا تستخدم الجوال في الشغل!")
                    phone_warning_given[found_person_id] = True
                phone_put_down_time[found_person_id] = None
            else:
                if found_person_id in phone_start_times:
                    if phone_put_down_time.get(found_person_id) is None:
                        phone_put_down_time[found_person_id] = current_time()
                    else:
                        elapsed = current_time() - phone_put_down_time[found_person_id]
                        if elapsed > 10:
                            phone_start_times.pop(found_person_id, None)
                            phone_warning_given.pop(found_person_id, None)
                            phone_put_down_time.pop(found_person_id, None)

        # Draw with cv2 instead of av
        for px1, py1, px2, py2 in detected_people:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        for phx1, phy1, phx2, phy2 in detected_phones:
            cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), (0, 0, 255), 2)

        cv2.imshow("الكاميرا - كشف الأشخاص والجوال", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    running.clear()
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()
