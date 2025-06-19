# main.py

from dotenv import load_dotenv
load_dotenv()

import os
import time
import threading
from pathlib import Path
import imaplib
import cv2
import time
from utils.audio import record_until_silence_fixed
from utils.fuzzy_wakeword import wait_for_wake_word, greeted_names, pending_greeting, GREETINGS, greet_detected_names
from api_clients.stt_api import transcribe_from_file
from api_clients.llm_api import classify_input, chat_response, generate_email_content
from api_clients.tts_api import speak, speak_warning
from api_clients.vision_api import detect_objects
from api_clients.tadawul import get_market_summary_report
from api_clients.email_api import send_email


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
    clear_all_tasks,
    get_all_tasks_for_display
)

SESSION_TIMEOUT = 60
session_active = False
last_interaction = 0

import wave
import audioop

SESSION_TIMEOUT = 60
session_active = False
last_interaction = 0

email_number_mapping = {}  # Global dictionary to store email number -> email data mapping

conversation_history = []
MAX_HISTORY_ITEMS = 5

def current_time():
    return time.time()

last_greet_time = 0
GREET_DELAY = 10  # seconds

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
        detected_names = detections.get("names", [])



        if detected_names:
            greet_detected_names(detected_names)

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
            silence_duration=3,
            max_duration=40
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
            pending_greeting = None
            print("💤 Session timed out.")
            speak("تم إنهاء الجلسة، ناديني إذا احتجتني!")

        return text.strip()

    except Exception as e:
        print(f"❌ Error during recording/transcription: {e}")
        return ""

# make a function to read last emails:
def read_last_emails(n=2):
    """
    Fetches the most recent n emails from the inbox and returns their subjects.
    """
    try:
        mail = imaplib.IMAP4_SSL(os.getenv("EMAIL_IMAP_SERVER"))
        mail.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        mail.select("inbox")

        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()[-n:]  # Get most recent n emails

        subjects = []
        for email_id in email_ids:
            result, msg_data = mail.fetch(email_id, "(BODY[HEADER.FIELDS (SUBJECT)])")
            subject = msg_data[0][1].decode().strip()
            subjects.append(subject)
        mail.close()
        mail.logout()
        return subjects

    except Exception as e:
        print(f"❌ Error fetching emails: {e}")
        speak("حصل خطأ في قراءة الإيميلات")
        return []

def read_email_contents(n=2, subject_title=None):

    
    try:
        import email
        from email.header import decode_header

        mail = imaplib.IMAP4_SSL(os.getenv("EMAIL_IMAP_SERVER"))
        mail.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        mail.select("inbox")

        if subject_title:
            # Search for emails with the specific subject
            search_criteria = f'SUBJECT "{subject_title}"'
            result, data = mail.search(None, search_criteria)
        else:
            # Get the most recent n emails
            result, data = mail.search(None, "ALL")
            data = [data[0].split()[-n:]]  # Get most recent n emails
        
        if not data[0]:
            print(f"No emails found with subject: {subject_title}")
            speak(f"ما لقيت أي إيميلات بعنوان '{subject_title}'")
            return []
            
        email_ids = data[0].split()
        
        emails = []
        for email_id in email_ids:
            result, msg_data = mail.fetch(email_id, "(RFC822)")  # Fetch the whole email
            raw_email = msg_data[0][1]
            
            # Parse the raw email
            msg = email.message_from_bytes(raw_email)
            
            # Get subject
            subject = msg["Subject"]
            if subject:
                subject, encoding = decode_header(subject)[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or 'utf-8', errors='replace')
            
            # Get body and count attachments
            body = ""
            attachment_count = 0
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    
                    # Count attachments
                    if "attachment" in content_disposition:
                        attachment_count += 1
                        print(f"📎 Found attachment: {part.get_filename()}")
                        continue
                    
                    # Get text content
                    if content_type == "text/plain" or content_type == "text/html":
                        try:
                            body_part = part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='replace')
                            body += body_part
                        except:
                            body += "Unable to decode this part of the message"
            else:
                # Not multipart - get the content directly
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='replace')
            
            # Announce the total number of attachments at the end
            if attachment_count > 0:
                print(f"📎 Total attachments: {attachment_count}")
                if attachment_count == 1:
                    speak(f"الإيميل يحتوي على مرفق واحد")
                elif attachment_count == 2:
                    speak(f"الإيميل يحتوي على مرفقين")
                elif attachment_count >= 3 and attachment_count <= 10:
                    speak(f"الإيميل يحتوي على {attachment_count} مرفقات")
                else:
                    speak(f"الإيميل يحتوي على {attachment_count} مرفق")
            
            emails.append({"subject": subject, "body": body, "attachment_count": attachment_count})
            
        mail.close()
        mail.logout()
        return emails

    except Exception as e:
        print(f"❌ Error fetching email contents: {e}")
        speak("حصل خطأ في قراءة محتوى الإيميلات")
        return []

def main():
    global session_active, index, embeddings, tasks
    print("🔰 Loading tasks...")
        # 🛡️ Load tasks safely with error handling
    try:
        load_tasks()
    except Exception as e:
        print(f"⚠️ Task loading failed: {e}. Starting with empty tasks.")
        # Initialize empty structures if loading fails
        from utils.task_handler import create_index
        global index, embeddings, tasks
        tasks = []
        embeddings = []
        index = create_index(dimension=384)

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

            if "ملخص السوق" in lower or "تقفيلة السوق" in lower or "سوق الأسهم" in lower:
                intent = "market_summary"
            if (
                "أخبار" in lower or 
                "خبر" in lower or 
                "آخر الأخبار" in lower or 
                "اخبار" in lower
            ):
                articles = fetch_okaz_headlines(n=5)
                news_summary = summarize_headlines(articles)
                print("📰 ملخص الأخبار:", news_summary)
                speak(news_summary)
                continue

            
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
                print(f"🔍 TASK RECORDING REQUEST: '{user_input}'")
                try:
                    add_task(user_input, conversation_history)
                    print("✅ Task recording completed")
                except Exception as e:
                    print(f"❌ Error in task recording: {e}")
                    import traceback
                    traceback.print_exc()
                    speak("حصل خطأ في تسجيل المهمة")

            # ───────────── sending Emails ─────────────

            elif intent == "ارسال_ايميل":
    # مثلا عشان نميز النوع بناء على الكلام
                if "مهام" in user_input:
                    email_body = generate_email_content("tasks", tasks)
                    subject = "قائمة المهام الخاصة بي"
                elif "تقديم" in user_input or "وظيفة" in user_input:
                    email_body = generate_email_content("job_application")
                    subject = "طلب تقديم وظيفة"
                else:
                    email_body = "ما فهمت نوع الإيميل اللي تبي ترسله."
                    speak(email_body)
                    continue

                send_email(subject, email_body)
                speak("تم إرسال الإيميل بنجاح.")

            # ───────────── Fetching Emails ─────────────
            
            
            # ───────────── Reading Topics ─────────────
            
            elif intent == "قراءة_ايميلات":
                print("🔍 Email reading request")
                try:
                    emails = read_last_emails(n=2)
                    if not emails:
                        speak("ما عندك إيميلات جديدة.")
                    else:
                        # Clear previous mapping and create a new one
                        global email_number_mapping
                        email_number_mapping = {}
                        
                        for i, subject in enumerate(emails, 1):
                            # Store the subject in our mapping
                            email_number_mapping[i] = {"subject": subject}
                            speak(f"الإيميل رقم {i}: {subject}")
                            
                        print(f"📧 Email mapping created: {email_number_mapping}")
                except Exception as e:
                    print(f"❌ Error reading emails: {e}")
                    speak("حصل خطأ في قراءة الإيميلات")
                    
            # ───────────── Reading the contents ─────────────     
                           
            elif intent == "قراءة_محتوى_ايميل":
                print("🔍 Email content reading request")
                try:
                    subject_title = None
                    email_number = None
                    
                    # Check if user is referring to an email by number
                    if "رقم" in user_input:
                        # Extract the number after "رقم"
                        number_parts = user_input.split("رقم")
                        if len(number_parts) > 1:
                            # Extract digits from the part after "رقم"
                            import re
                            num_match = re.search(r'\d+', number_parts[1])
                            if num_match:
                                email_number = int(num_match.group())
                                print(f"🔍 User requested email number {email_number}")
                                
                                # Check if this number exists in our mapping
                                if email_number in email_number_mapping:
                                    subject_title = email_number_mapping[email_number].get("subject")
                                    print(f"📧 Found subject: {subject_title}")
                                else:
                                    speak(f"لم أجد إيميل برقم {email_number}")
                                    return
        
                    # If not by number, check for subject title
                    elif "بعنوان" in user_input:
                        subject_title = user_input.split("بعنوان")[-1].strip()
                    
                    # Get email contents
                    emails = read_email_contents(n=2, subject_title=subject_title)
                    if not emails:
                        speak("ما لقيت أي إيميلات جديدة.")
                    else:
                        # Update mapping with full content
                        email_number_mapping = {}
                        
                        for i, email in enumerate(emails, 1):
                            # Store the complete email in our mapping
                            email_number_mapping[i] = email
                            
                            speak(f"الإيميل رقم {i} بعنوان: {email['subject']}")
                            speak(f"محتوى الإيميل: {email['body']}")
                            if email['attachment_count'] > 0:
                                speak(f"هذا الإيميل يحتوي على {email['attachment_count']} مرفق.")
                except Exception as e:
                    print(f"❌ Error reading email contents: {e}")
                    speak("حصل خطأ في قراءة محتوى الإيميلات")
            # ───────────── Task related stuff ─────────────
            elif intent == "تذكير":
                print("🔍 Task listing request")
                try:
                    # Get all tasks directly from JSON
                    all_task_texts = get_all_tasks_for_display()
                    
                    if all_task_texts:
                        if len(all_task_texts) == 1:
                            speak(f"عندك مهمة واحدة: {all_task_texts[0]}")
                        else:
                            tasks_text = "، ".join(all_task_texts)
                            speak(f"مهامك الحالية هي: {tasks_text}")
                    else:
                        speak("ما عندك مهام مسجلة حالياً.")
                        
                except Exception as e:
                    print(f"❌ Error listing tasks: {e}")
                    speak("حصل خطأ في جلب المهام")

            elif intent == "حذف":
                delete_task(user_input)

            # ───────────── Market summary ─────────────
            
            elif intent == "ملخص_السوق":
                response_text = get_market_summary_report()
                speak(response_text)  # Your TTS function
            else:
                response = chat_response(user_input, [])
                print("🤖", response)
                speak(response)
                # added context awareness:
                conversation_history.append({"user": user_input, "assistant": response})
                
                if len(conversation_history) > MAX_HISTORY_ITEMS:
                    conversation_history.pop(0)
                    
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
