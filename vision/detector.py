import cv2
import torch
import time

print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5
print("YOLOv5 loaded successfully!")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No camera found!")
    exit()

greeting_threshold = 0.7
last_greeting_time = 0
greeting_cooldown = 3.0

unique_people = {}
person_id_counter = 0
phone_times = {}
phone_start_times = {}
phone_warning_given = {}
phone_put_down_time = {}  # Track when phone was put down  # Track if warning was given for current phone session

PERSON_CLASS = 0
CELL_PHONE_CLASS = 67

def assign_person_id(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    box_width = x2 - x1
    box_height = y2 - y1
    current_time = time.time()
    
    # Check existing people
    for person_id in unique_people:
        ex, ey, last_seen = unique_people[person_id]
        distance = ((center_x - ex) ** 2 + (center_y - ey) ** 2) ** 0.5
        
        # Large threshold for better tracking
        threshold = max(250, (box_width + box_height) // 2)
        
        if distance < threshold:
            unique_people[person_id] = (center_x, center_y, current_time)
            return person_id
    
    # Create new person
    global person_id_counter
    person_id_counter += 1
    unique_people[person_id_counter] = (center_x, center_y, current_time)
    phone_times[person_id_counter] = 0
    phone_start_times[person_id_counter] = None
    phone_warning_given[person_id_counter] = False
    phone_put_down_time[person_id_counter] = None
    print(f"New person detected! ID: {person_id_counter}")
    return person_id_counter

def cleanup_old_people():
    current_time = time.time()
    to_remove = []
    
    for person_id in unique_people:
        x, y, last_seen = unique_people[person_id]
        if current_time - last_seen > 20:
            to_remove.append(person_id)
    
    for person_id in to_remove:
        del unique_people[person_id]

def check_phone_detection(person_box, all_detections):
    px1, py1, px2, py2 = person_box
    
    for *box, conf, cls in all_detections:
        if int(cls) == CELL_PHONE_CLASS and conf > 0.3:  # Lower confidence threshold
            x1, y1, x2, y2 = map(int, box)
            
            phone_width = x2 - x1
            phone_height = y2 - y1
            
            # More flexible size limits
            if phone_width < 15 or phone_height < 15:  # Smaller minimum
                continue
            if phone_width > 200 or phone_height > 300:  # Larger maximum
                continue
                
            # More flexible aspect ratio - allows horizontal phones too
            aspect_ratio = max(phone_height, phone_width) / min(phone_height, phone_width)
            if aspect_ratio > 4.0:  # Very flexible ratio
                continue
            
            # More flexible positioning - phone can be partially outside person box
            phone_center_x = (x1 + x2) // 2
            phone_center_y = (y1 + y2) // 2
            
            # Check if phone center is near person box (more forgiving)
            if (px1 - 100 <= phone_center_x <= px2 + 100 and 
                py1 - 100 <= phone_center_y <= py2 + 100):
                return True, conf, (x1, y1, x2, y2)
    
    return False, 0, None

# Calculate session duration
session_start_time = time.time()

print("Human Detection Started!")
print("Also detecting phones")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_time = time.time()
    
    cleanup_old_people()
    
    current_detections = []
    
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == PERSON_CLASS and conf > greeting_threshold:
            x1, y1, x2, y2 = map(int, box)
            
            person_id = assign_person_id(x1, y1, x2, y2)
            
            has_phone, phone_conf, phone_box = check_phone_detection((x1, y1, x2, y2), results.xyxy[0])
            
            # Track phone holding time
            if has_phone:
                if person_id not in phone_start_times:
                    phone_start_times[person_id] = current_time
                elif phone_start_times[person_id] is None:
                    phone_start_times[person_id] = current_time
                    # Check if enough time passed since putting down phone
                    put_down_time = phone_put_down_time.get(person_id, None)
                    if put_down_time is None or (current_time - put_down_time) >= 5:
                        phone_warning_given[person_id] = False  # Allow new warning
                
                # Give warning only once per phone session (after 5 second break)
                if not phone_warning_given.get(person_id, False):
                    message = "لو سمحت، اترك الجوال!"
                    print(message)
                    with open("phone_warnings.txt", "a", encoding="utf-8") as f:
                        f.write(f"{message}\n")
                    phone_warning_given[person_id] = True
                    
            else:
                if person_id in phone_start_times and phone_start_times[person_id] is not None:
                    session_duration = current_time - phone_start_times[person_id]
                    if person_id not in phone_times:
                        phone_times[person_id] = 0
                    phone_times[person_id] += session_duration
                    phone_start_times[person_id] = None
                    phone_put_down_time[person_id] = current_time  # Record when phone was put down
            
            if (current_time - last_greeting_time) > greeting_cooldown:
                if not has_phone:
                    print(f"Hello there person #{person_id}!")
                    last_greeting_time = current_time
            
            color = (0, 255, 255) if has_phone else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if has_phone and phone_box:
                px1, py1, px2, py2 = phone_box
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(frame, f'Phone: {phone_conf:.1%}', (px1, py1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            label = f'Person #{person_id}: {conf:.1%}'
            if has_phone:
                label += ' [PHONE]'
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            current_detections.append(person_id)
    
    unique_count = len(unique_people)
    current_count = len(current_detections)
    
    info_text = f'Current: {current_count} | Total Unique: {unique_count}'
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    cv2.imshow('Human + Phone Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final phone time calculation
total_session_time = time.time() - session_start_time
total_phone_time = 0

for person_id in phone_start_times:
    if phone_start_times[person_id] is not None:
        final_session = time.time() - phone_start_times[person_id]
        if person_id not in phone_times:
            phone_times[person_id] = 0
        phone_times[person_id] += final_session

# Calculate total phone usage across all people
for person_id in phone_times:
    total_phone_time += phone_times[person_id]

# Calculate percentage
phone_percentage = (total_phone_time / total_session_time * 100) if total_session_time > 0 else 0

print(f"\nFinal Summary:")
print(f"Unique persons detected: {len(unique_people)}")
print("Phone usage statistics:")
for person_id in unique_people:
    total_time = phone_times.get(person_id, 0)
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    if minutes > 0:
        print(f"   Person #{person_id}: {minutes}m {seconds}s with phone")
    else:
        print(f"   Person #{person_id}: {seconds}s with phone")

percentage_message = f"استخدمت جوالك {phone_percentage:.1f}% من الوقت"
print(percentage_message)

# Save percentage to file
with open("phone_warnings.txt", "a", encoding="utf-8") as f:
    f.write(f"{percentage_message}\n")

print("Detection stopped!")