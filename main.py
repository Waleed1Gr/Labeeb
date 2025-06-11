from tasks.task_manager import (
    load_tasks,
    add_task,
    search_tasks,
    delete_task,
    classify_input,
    chat_response,
)
from utils.config import session_active, TEMP_DIR
from audio.speak import (
    speak,
    stop_current_speech,
    is_currently_speaking,
    generate_speech_bytes,
)
from audio.listen import (
    record_and_transcribe,
    wait_for_wake_word,
    transcribe_audio_bytes,
)
import time
import threading
from vision.camera import phone_person_detector
from vision.detector import detect_from_image_bytes
from utils.config import client, session_active
import shutil
import asyncio
import websockets
import json
import base64


async def handle_pi(websocket):
    print("🔗 Pi connected!")
    load_tasks()
    try:
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "audio":
                audio_bytes = base64.b64decode(data["data"])
                text = transcribe_audio_bytes(audio_bytes)
                # Process intent, etc.
                intent = classify_input(text)
                if intent == "تسجيل":
                    add_task(text)
                    reply = "تم تسجيل المهمة!"
                elif intent == "تذكير":
                    related = search_tasks(text)
                    reply = chat_response(text, related)
                else:
                    reply = "تم الاستلام: " + text
                tts_bytes = generate_speech_bytes(reply)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "tts",
                            "audio": base64.b64encode(tts_bytes).decode(),
                            "text": reply,
                        }
                    )
                )
            elif data["type"] == "camera":
                img_bytes = base64.b64decode(data["data"])
                detections = detect_from_image_bytes(img_bytes)
                await websocket.send(
                    json.dumps({"type": "detection", "detections": detections})
                )
    except Exception as e:
        print(f"Connection error: {e}")


def main():
    try:
        # Ensure temp directory exists
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        print("🔰 جاري تحميل المهام...")
        load_tasks()
        print("🚀 تم تشغيل البرنامج بنجاح!")

        while True:
            try:
                if not session_active:
                    # Only try to detect wake word when session is inactive
                    if wait_for_wake_word():
                        session_active = True
                    continue

                user_input = record_and_transcribe(
                    wait_for_wake=False
                )  # Important: don't check wake word here
                if not user_input:
                    continue

                # Process normal input
                intent = classify_input(user_input)

                if is_currently_speaking():
                    stop_current_speech()
                    time.sleep(0.1)

                if intent == "تسجيل":
                    add_task(user_input)
                elif intent == "تذكير":
                    related = search_tasks(user_input)
                    response = chat_response(user_input, related)
                    print("🤖", response)
                    speak(response)
                    while is_currently_speaking():
                        time.sleep(0.1)
                    if "<close_conversation>" in response:
                        session_active = False
                        print("\n👋 انتهت المحادثة")
                        print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
                        continue
                else:
                    try:
                        prompt = f"""أنت مساعد شخصي باللهجة السعودية، رد على السؤال التالي:
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
                        while is_currently_speaking():
                            time.sleep(0.1)
                        if "<close_conversation>" in reply:
                            session_active = False
                            print("\n👋 انتهت المحادثة")
                            print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
                            continue
                    except Exception as e:
                        print(f"General chat error: {e}")
                        speak("حصل خطأ في الرد، حاول مرة ثانية")
                        while is_currently_speaking():
                            time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n🛑 إيقاف البرنامج...")
                break
            except Exception as e:
                print(f"Main loop error: {e}")
                continue

    finally:
        # Cleanup
        try:
            stop_current_speech()
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    start_server = websockets.serve(handle_pi, "0.0.0.0", 6789)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
    main()
