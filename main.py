from tasks.task_manager import load_tasks, add_task, search_tasks, delete_task, classify_input, chat_response
from utils.config import session_active, TEMP_DIR
from audio.speak import speak, stop_current_speech, is_currently_speaking
from audio.listen import record_and_transcribe, wait_for_wake_word
import time
import threading
from vision.camera import phone_person_detector
from utils.config import client
import shutil

def main():
    try:
        # Ensure temp directory exists
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        print("🔰 جاري تحميل المهام...")
        load_tasks()
        print("🚀 تم تشغيل البرنامج بنجاح!")
        
        while True:
            try:
                user_input = record_and_transcribe()
                if not user_input:
                    continue
                
                # Check for exit phrases
                exit_phrases = ["مع السلامة", "خلاص", "باي", "وقف"]
                if any(phrase in user_input.lower() for phrase in exit_phrases):
                    response = "مع السلامة يا طيب، إذا احتجت شي ثاني ناديني.\n<close_conversation>"
                    print("🤖", response)
                    speak(response)
                    session_active = False
                    print("\n👋 انتهت المحادثة")
                    print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
                    continue
                
                # Process normal input
                intent = classify_input(user_input)
                
                if is_currently_speaking():
                    stop_current_speech()
                    time.sleep(0.1)  # Give time for cleanup
                
                if intent == "تسجيل":
                    add_task(user_input)
                elif intent == "تذكير":
                    related = search_tasks(user_input)
                    response = chat_response(user_input, related)
                    print("🤖", response)
                    speak(response)
                    if "<close_conversation>" in response:
                        session_active = False
                        print("\n👋 انتهت المحادثة")
                        print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
                        continue
                elif intent == "حذف":
                    delete_task(user_input)
                else:
                    try:
                        prompt = f"""أنت مساعد شخصي باللهجة السعودية، رد على السؤال التالي:
                        السؤال: {user_input}
                        """
                        res = client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        )
                        reply = res.choices[0].message.content.strip()
                        print("🤖", reply)
                        speak(reply)
                        if "<close_conversation>" in reply:
                            print("\n👋 انتهت المحادثة")
                            session_active = False
                            print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
                            continue
                    except Exception as e:
                        print(f"General chat error: {e}")
                        speak("حصل خطأ في الرد، حاول مرة ثانية")
                    
            except KeyboardInterrupt:
                print("\n🛑 إيقاف البرنامج...")
                break
            except Exception as e:
                print(f"Main loop error: {e}")
                continue

    finally:
        # Cleanup
        try:
            stop_current_speech()  # Stop any ongoing speech
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()