import time
import cv2
import torch
from utils.utils import current_time, speak_warning


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
    """Enhanced phone/person detector with headless operation"""
    try:
        print("📸 محاولة تحميل نموذج YOLOv5...")
        
        # Try multiple approaches to load YOLOv5
        model = None
        
        # Method 1: Try ultralytics package directly
        try:
            from ultralytics import YOLO
            #TODO: replace the yolo5 model with the yolo8 model.
            model = YOLO('yolov5su.pt')
            print("✅ تم تحميل YOLOv5 من ultralytics")
        except ImportError:
            # here, there was another method in case yolo didn'd load, it was by loading it from a local repo. deprecated.
            print("❌ لم أستطع تحميل نموذج YOLOv5، سيتم تشغيل البرنامج بدون كشف الجوال")
            return

        # Initialize camera
        print("📸 محاولة تشغيل الكاميرا...")
        
        # Try different camera backends for macOS
        backends_to_try = [
            cv2.CAP_AVFOUNDATION,  # macOS default
            cv2.CAP_V4L2,          # Linux
            cv2.CAP_DSHOW,         # Windows
            cv2.CAP_ANY            # Auto-detect
        ]
        
        cap = None
        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    print(f"✅ تم تشغيل الكاميرا بنجاح مع backend: {backend}")
                    break
                else:
                    cap.release()
            except:
                continue
        
        if cap is None or not cap.isOpened():
            print("❌ الكاميرا غير متوفرة، سيتم تشغيل البرنامج بدون كشف الجوال")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
        
        print("✅ تم تشغيل الكاميرا بنجاح - يعمل في الخلفية بدون عرض")
        
        # Initialize tracking variables
        unique_people = {}
        person_id_counter = 0
        phone_start_times = {}
        phone_warning_given = {}
        phone_put_down_time = {}
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ لم أستطع التقاط صورة من الكاميرا")
                time.sleep(1)
                continue

            frame_count += 1
            
            # Process every 3rd frame to reduce CPU usage
            if frame_count % 3 != 0:
                continue

            try:
                # Process frame with model
                if hasattr(model, 'predict'):  # ultralytics YOLO
                    results = model.predict(frame, verbose=False)
                    # Process ultralytics results
                    detected_people = []
                    detected_phones = []
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                if conf > 0.5:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    if cls == 0:  # person
                                        detected_people.append((int(x1), int(y1), int(x2), int(y2)))
                                    elif cls == 67:  # cell phone
                                        detected_phones.append((int(x1), int(y1), int(x2), int(y2)))
                else:  # torch.hub YOLOv5
                    results = model(frame)
                    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
                    
                    detected_people = []
                    detected_phones = []
                    
                    for i, label in enumerate(labels):
                        if int(label) == 0:  # person
                            x1, y1, x2, y2, _ = cords[i]
                            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                            detected_people.append((x1, y1, x2, y2))
                        elif int(label) == 67:  # cell phone
                            x1, y1, x2, y2, _ = cords[i]
                            x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                            detected_phones.append((x1, y1, x2, y2))

                # Log detections periodically
                if frame_count % 150 == 0:  # Every ~10 seconds at 15 FPS
                    print(f"🔍 كشف: {len(detected_people)} أشخاص، {len(detected_phones)} جوالات")

                # Process detections (same logic as before)
                for px1, py1, px2, py2 in detected_people:
                    person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                    min_dist = float('inf')
                    closest_phone = None

                    for phx1, phy1, phx2, phy2 in detected_phones:
                        phone_center = ((phx1 + phx2) // 2, (phy1 + phy2) // 2)
                        dist = ((person_center[0] - phone_center[0]) ** 2 + (person_center[1] - phone_center[1]) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_phone = (phx1, phy1, phx2, phy2)

                    # Person tracking
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

                    # Phone usage tracking
                    if closest_phone and min_dist < 150:
                        now = current_time()
                        if found_person_id not in phone_start_times:
                            phone_start_times[found_person_id] = now
                            phone_warning_given[found_person_id] = False
                            print(f"📱 شخص رقم {found_person_id} بدأ استخدام الجوال")

                        duration = now - phone_start_times[found_person_id]
                        if duration > 10 and not phone_warning_given[found_person_id]:
                            print(f"⚠️ تحذير للشخص رقم {found_person_id} - استخدم الجوال لأكثر من 10 ثواني")
                            speak_warning("يالحبيب، لا تستخدم الجوال في الشغل!")
                            phone_warning_given[found_person_id] = True
                        phone_put_down_time[found_person_id] = None
                    else:
                        if found_person_id in phone_start_times:
                            if phone_put_down_time.get(found_person_id) is None:
                                phone_put_down_time[found_person_id] = current_time()
                                print(f"📵 شخص رقم {found_person_id} ترك الجوال")
                            else:
                                elapsed = current_time() - phone_put_down_time[found_person_id]
                                if elapsed > 10:
                                    phone_start_times.pop(found_person_id, None)
                                    phone_warning_given.pop(found_person_id, None)
                                    phone_put_down_time.pop(found_person_id, None)

                # Save frame with detections occasionally for debugging
                if frame_count % 300 == 0:  # Every ~20 seconds
                    debug_frame = frame.copy()
                    for px1, py1, px2, py2 in detected_people:
                        cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    for phx1, phy1, phx2, phy2 in detected_phones:
                        cv2.rectangle(debug_frame, (phx1, phy1), (phx2, phy2), (0, 0, 255), 2)
                    cv2.imwrite(f"detection_frame_{int(time.time())}.jpg", debug_frame)
                
                time.sleep(0.1)  # Prevent high CPU usage
                    
            except Exception as processing_error:
                print(f"Frame processing error: {processing_error}")
                continue

        cap.release()
        
    except Exception as e:
        print(f"Camera detector error: {e}")
        print("⚠️ سيتم تشغيل البرنامج بدون كشف الجوال")

