# main.py

from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from pathlib import Path

import cv2
import time
from utils.audio import record_until_silence_fixed
from utils.fuzzy_wakeword import wait_for_wake_word
from api_clients.stt_api import transcribe_from_file
from api_clients.llm_api import classify_input, chat_response
from api_clients.tts_api import speak, speak_warning
from api_clients.vision_api import detect_objects

# ─── Import the Okaz RSS functions ───────────────────────────────────────────────
from api_clients.news_api import fetch_okaz_headlines, summarize_headlines

from utils.task_handler import (
    load_tasks,
    add_task,
    delete_task,
    search_tasks,
    save_tasks,
    tasks,
    embeddings,
    index,
    create_index,
    clear_all_tasks
)

SESSION_TIMEOUT = 60
session_active = False
last_interaction = 0

import wave
import audioop

SESSION_TIMEOUT = 60
session_active = False
last_interaction = 0

def current_time():
    return time.time()

def phone_person_detector():
    """
    Continuously monitors webcam frames for a phone being held by a person.
    If a phone overlaps with a person for 10+ seconds, gives a spoken warning.
    This runs in a background thread, independent of wake word.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not available; vision disabled.")
        return

    print("📷 Vision thread started (local YOLO)")

    phone_timer_start = 0
    warning_issued = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        detections = detect_objects(frame)
        people_boxes = detections.get("people", [])
        phone_boxes = detections.get("phones", [])

        # print(f"🔍 Detected {len(people_boxes)} people, {len(phone_boxes)} phones")

        # Check for overlap: is any phone inside a person box?
        overlap = False
        for (fx1, fy1, fx2, fy2) in phone_boxes:
            phone_cx = (fx1 + fx2) // 2
            phone_cy = (fy1 + fy2) // 2
            for (px1, py1, px2, py2) in people_boxes:
                if px1 < phone_cx < px2 and py1 < phone_cy < py2:
                    overlap = True
                    break
            if overlap:
                break

        if overlap:
            if phone_timer_start == 0:
                phone_timer_start = time.time()
                print("⏱️ Started phone timer")

            elapsed = time.time() - phone_timer_start
            # print(f"⏱️ Phone overlap duration: {elapsed:.1f}s")

            if elapsed >= 10.0 and not warning_issued:
                print("📢 Warning issued: phone held over 10 seconds")
                speak_warning("لو سمحت لاتستخدم الجوال في الشغل")
                warning_issued = True
        else:
            if phone_timer_start != 0:
                print("🔄 Overlap ended; resetting phone timer")
            phone_timer_start = 0
            warning_issued = False

        # Slight sleep to reduce CPU load
        time.sleep(0.1)
def current_time():
    return time.time()

# 🎙️ Main STT handler
def record_and_transcribe(wait_for_wake: bool = True) -> str:
    global session_active, last_interaction

    if wait_for_wake and not session_active:
        if not wait_for_wake_word():
            return ""
        session_active = True
        last_interaction = current_time()

    filename = Path(f"record_{int(time.time())}.wav")

    try:
        print("🎙️ Recording audio...")
        success = record_until_silence_fixed(
            filename,
            sample_rate=16000,
            silence_duration=2,
            max_duration=10
        )

        if not success:
            print("⚠️ No speech detected.")
            return ""

        # 🧠 Transcribe
        text = transcribe_from_file(
            filename,
            language="ar",
            prompt="توقع كلام باللهجة السعودية"
        )

        # Clean up the audio file
        try:
            filename.unlink()
        except:
            pass

        # 🧹 Check STT junk
        junk_phrases = {"اه", "ام", "مم", "اا", "مممم", "اهم", "اااا", ""}
        if not text or text.strip() in junk_phrases:
            print("🧹 Junk or empty phrase — skipping.")
            return ""

        print("📄 Transcribed text:", text)

        # ⏳ Update session activity
        last_interaction = current_time()

        if session_active and (current_time() - last_interaction > SESSION_TIMEOUT):
            session_active = False
            print("💤 Session timed out.")
            speak("تم إنهاء الجلسة، ناديني إذا احتجتني!")

        return text.strip()

    except Exception as e:
        print(f"❌ Error during recording/transcription: {e}")
        return ""

def main():
    global session_active, index, embeddings, tasks
    print("🔰 Loading tasks...")
    load_tasks()

    # Start vision thread
    cam_thread = threading.Thread(target=phone_person_detector, daemon=True)
    cam_thread.start()

    print("🚀 Assistant is up and running!")

    while True:
        try:
            user_input = record_and_transcribe(wait_for_wake=not session_active)
            if not user_input:
                if session_active:
                    session_active = False
                    print("\n💤 Session ended.")
                    speak("تم إنهاء الجلسة، ناديني إذا احتجتني!")
                continue

            lower = user_input.strip().lower()

            # 🔍 Optional: keyword override for حذف الكل
            if "احذف كل" in lower or "احذف المهام كلها" in lower:
                intent = "حذف_الكل"
            else:
                intent = classify_input(user_input)

            # ───────────── Intent: Delete All Tasks ─────────────
            if intent == "حذف_الكل":
                speak("متأكد تبيني أحذف كل المهام؟ قل نعم أو لا.")
                confirmation = ""
                while confirmation not in ["نعم", "لا"]:
                    confirmation = record_and_transcribe(wait_for_wake=False).strip().lower()

                if confirmation == "نعم":
                    clear_all_tasks()
                    speak("تمام، مسحت كل المهام.")
                else:
                    speak("ما صار شيء، ما مسحت ولا مهمة.")
                continue

            elif intent == "حذف":
                delete_task(user_input)
            # ───────────── Other intents ─────────────
            elif intent == "تسجيل":
                add_task(user_input)

            elif intent == "تذكير":
                related = search_tasks(user_input)
                if not related:
                    if tasks:
                        all_texts = "، ".join([t["text"] for t in tasks])
                        speak(f"مهامك الحالية هي: {all_texts}")
                    else:
                        speak("ما عندك مهام مسجلة حالياً.")
                else:
                    response = chat_response(user_input, related)
                    print("🤖", response)
                    speak(response)
                    if "<close_conversation>" in response:
                        session_active = False
                        print("🔊 Waiting for wake word: 'لبيب'...")
                        continue

            elif intent == "حذف":
                delete_task(user_input)

            else:
                response = chat_response(user_input, [])
                print("🤖", response)
                speak(response)
                if "<close_conversation>" in response:
                    session_active = False
                    print("🔊 Waiting for wake word: 'لبيب'...")
                    continue

        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            speak("حصل خطأ، بس راح أكمل شغل")


if __name__ == "__main__":
    main()
