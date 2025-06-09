from dotenv import load_dotenv
load_dotenv()

import time
from pathlib import Path
from api_clients.stt_api import transcribe_from_file
from api_clients.tts_api import speak
import sounddevice as sd
from scipy.io.wavfile import write  # to write proper WAV files
import numpy as np

# sd.default.device = (2, None)

WAKE_WORDS = ["لبيب", "لبي", "لب", "labeeb"]
SILENCE_THRESHOLD = 80  # Adjust this threshold based on microphone sensitivity

def wait_for_wake_word(duration: int = 3, sample_rate: int = 16000) -> bool:
    """
    Continuously records `duration` seconds of audio, checks for silence, writes a proper WAV header,
    sends it to the STT API, and checks if any of the WAKE_WORDS appear in the transcript.
    Returns True as soon as a wake word is detected.
    """
    print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
    while True:
        try:
            # Record raw audio data (int16)
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()
            amplitude = np.abs(recording).mean()
            print("متوسط الصوت:", amplitude)

            # Check for silence: compute mean absolute amplitude
            audio_data = recording.flatten()
            if np.mean(np.abs(audio_data)) < SILENCE_THRESHOLD:
                # Detected silence (below threshold), skip STT and retry
                time.sleep(0.1)
                continue

            # Save as proper WAV file (overwrites existing)
            write("wake_check.wav", sample_rate, recording)

            # Transcribe only if not silent
            text = transcribe_from_file(
                Path("wake_check.wav"),
                language="ar",
                prompt="لبيب هو اسم الروبوت. الكلمات المتوقعة: لبيب"
            )
            if(amplitude < 0.002):
                print("🔇 لا يوجد صوت كافي، إعادة المحاولة...")
            else:
                print("👂 سمع:", text)

            if any(word in text for word in WAKE_WORDS):
                print("✨ تم التعرف على كلمة التنبيه!")
                speak("نعم، كيف اقدر اخدمك؟")
                return True

            time.sleep(0.1)
        except Exception as e:
            print(f"❌ خطأ في التعرف على الصوت: {e}")
            time.sleep(1)
            continue
