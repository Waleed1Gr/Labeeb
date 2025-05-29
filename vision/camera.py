import cv2
import torch
import time
from audio.speak import speak_warning


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
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.conf = 0.5
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ الكاميرا غير متوفرة")
        return

    unique_people = {}
    person_id_counter = 0
    phone_start_times = {}
    phone_warning_given = {}
    phone_put_down_time = {}

    def current_time():
        return time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        detected_people = []
        detected_phones = []

        for i, label in enumerate(labels):
            if int(label) == 0:  # شخص
                x1, y1, x2, y2, _ = cords[i]
                x1, y1, x2, y2 = (
                    int(x1 * frame.shape[1]),
                    int(y1 * frame.shape[0]),
                    int(x2 * frame.shape[1]),
                    int(y2 * frame.shape[0]),
                )
                detected_people.append((x1, y1, x2, y2))
            elif int(label) == 67:  # جوال
                x1, y1, x2, y2, _ = cords[i]
                x1, y1, x2, y2 = (
                    int(x1 * frame.shape[1]),
                    int(y1 * frame.shape[0]),
                    int(x2 * frame.shape[1]),
                    int(y2 * frame.shape[0]),
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

        for px1, py1, px2, py2 in detected_people:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
        for phx1, phy1, phx2, phy2 in detected_phones:
            cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), (0, 0, 255), 2)

        cv2.imshow("الكاميرا - كشف الأشخاص والجوال", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
