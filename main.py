import os

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
from tasks.task_manager import (
    load_tasks,
    add_task,
    delete_task,
    search_tasks,
    chat_response,
    classify_input,
)
from utils.config import client
from audio.listen import record_and_transcribe, speak
from vision.camera import phone_person_detector
import threading
import time


# ============== MAIN ==============
def main():
    print("🔰 جاري تحميل المهام...")
    load_tasks()

    # شغل الكاميرا في ثريد منفصل
    cam_thread = threading.Thread(target=phone_person_detector, daemon=True)
    cam_thread.start()

    time.sleep(2)  # أعطي الكاميرا وقت تشتغل

    while True:
        user_input = record_and_transcribe()
        if not user_input:
            speak("ما سمعتك، حاول مرة ثانية.")
            continue

        intent = classify_input(user_input)
        # print(f"🎯 التصنيف: {intent}")

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
            # ✅ استخدم ChatGPT للرد على استفسارات عامة بصيغة لطيفة
            prompt = f"""أنت مساعد شخصي باللهجة السعودية، رد على السؤال التالي حتى لو ما كان تسجيل أو تذكير أو حذف.
            رد مختصر، واضح، خفيف دم:

            السؤال: {user_input}
        """
            res = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            reply = res.choices[0].message.content.strip()
            print("🤖", reply)
            speak(reply)


if __name__ == "__main__":
    main()
