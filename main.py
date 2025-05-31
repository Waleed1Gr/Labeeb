from tasks.task_manager import load_tasks, add_task, search_tasks, delete_task, classify_input, chat_response
from utils.config import session_active
from audio.speak import speak
from audio.listen import record_and_transcribe
import time
import threading
from vision.camera import phone_person_detector
from utils.config import client

def main():
    print("🔰 جاري تحميل المهام...")
    load_tasks()

    # Start camera in separate thread
    cam_thread = threading.Thread(target=phone_person_detector, daemon=True)
    cam_thread.start()

    time.sleep(2)  # Give camera time to start

    print("🚀 تم تشغيل البرنامج بنجاح!")
    
    while True:
        try:
            user_input = record_and_transcribe()
            if not user_input:
                speak("ما سمعتك، حاول مرة ثانية.")
                continue

            intent = classify_input(user_input)
            
            if intent == "تسجيل":
                add_task(user_input)
            elif intent == "تذكير":
                related = search_tasks(user_input)
                response = chat_response(user_input, related)
                print("🤖", response)
                speak(response)
            elif intent == "حذف":
                delete_task(user_input)
            else:
                # Use ChatGPT for general queries
                try:
                    prompt = f"""أنت مساعد شخصي باللهجة السعودية، رد على السؤال التالي حتى لو ما كان تسجيل أو تذكير أو حذف.
                    رد مختصر، واضح، خفيف دم:

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
                except Exception as e:
                    print(f"General chat error: {e}")
                    speak("حصل خطأ في الرد، حاول مرة ثانية")
                    
        except KeyboardInterrupt:
            print("🛑 إيقاف البرنامج...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            speak("حصل خطأ، بس راح أكمل شغل")

if __name__ == "__main__":
    main()