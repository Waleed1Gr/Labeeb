# listen.py
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import time
from utils.config import wake_word
from models.models import whisper_model
import difflib
from audio.speak import speak


# ============== WAKE WORD & RECORD ==============
def wait_for_wake_word():
    fs = 16000
    duration = 5  # زودنا المدة شوي عشان يقل الخطأ
    print(f"🔊 بانتظار كلمة التنبيه: '{wake_word}'...")

    while True:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("wake_check.wav", fs, recording)

        result = whisper_model.transcribe("wake_check.wav", language="ar")
        text = result["text"].strip().lower()
        print("👂 سمع:", text)

        # ✅ fuzzy matching بدل التطابق الحرفي
        matches = difflib.get_close_matches(wake_word, [text], n=1, cutoff=0.7)
        if matches:
            print("🚨 تم التعرف على كلمة قريبة من 'لبيب'!")
            speak("كيف اقدر اخدمك؟")
            return


def record_and_transcribe():
    wait_for_wake_word()

    # ✅ إضافة صوت تنبيه هنا

    fs = 16000
    seconds = 9
    print("🎙️ تسجيل الآن...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("recorded.wav", fs, recording)
    print("✅ تم التسجيل")

    result = whisper_model.transcribe("recorded.wav", language="ar")
    text = result["text"].strip()
    print("📄 النص:", text)
    return text
