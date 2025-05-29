from pathlib import Path
import os
import time
from pydub import AudioSegment
from openai import OpenAI
from utils.config import client


# ============== SPEAK FUNCTION ==============
def speak(text):
    filename = f"response_{int(time.time())}.mp3"
    path = Path(filename)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions="تكلم بلهجة سعودية نجدية واضحة، لا تغلط بالكلمات وخلك ذكي",
    ) as response:
        response.stream_to_file(path)

    # احصل على مدة الصوت
    audio = AudioSegment.from_file(path)
    duration_sec = len(audio) / 1000  # بالملي ثانية → ثانية

    # شغل الصوت
    if os.name == "nt":
        os.system(f'start /min /wait "" "{path}"')  # يشغل وينتظر في ويندوز
    else:
        os.system(f"afplay {path}")  # macOS يشغل تلقائياً وينتظر

    # انتظر مدة الصوت قبل الرجوع
    time.sleep(duration_sec)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ash",
        input=text,
        instructions="تكلم بلهجة سعودية نجدية واضحة، لا تغلط بالكلمات وخلك ذكي",
    ) as response:
        response.stream_to_file(path)

    if os.name == "nt":
        os.system(f'start "" "{path}"')
    else:
        os.system(f"afplay {path}")
