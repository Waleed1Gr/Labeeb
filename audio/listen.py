import time
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad
from pydub import AudioSegment
import queue
import collections
from pathlib import Path
from utils.config import SESSION_TIMEOUT, last_interaction, session_active, TEMP_DIR
from utils.utils import current_time
from audio.speak import speak, stop_current_speech, is_currently_speaking
from models.model import whisper_model

# Initialize VAD with moderate sensitivity
vad = webrtcvad.Vad(2)


def record_until_silence(
    filename: str,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    silence_duration: int = 2,
    max_duration: int = 10,
) -> bool:
    """Records audio until silence is detected"""
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate, channels=1, dtype="int16", callback=callback
    )
    stream.start()

    ring_buffer = collections.deque()
    silence_start = None
    start_time = time.time()
    buffer = b""

    frame_size = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_frame = frame_size * 2
    speech_detected = False

    try:
        while True:
            if time.time() - start_time > max_duration:
                print("⏰ Max duration reached.")
                break

            try:
                data = q.get(timeout=1)
            except queue.Empty:
                continue

            buffer += data.tobytes()
            while len(buffer) >= bytes_per_frame:
                frame = buffer[:bytes_per_frame]
                buffer = buffer[bytes_per_frame:]

                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    silence_start = None
                    speech_detected = True
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("🔇 Silence detected, stopping recording.")
                        stream.stop()
                        stream.close()
                        audio_data = np.frombuffer(
                            b"".join(ring_buffer), dtype=np.int16
                        )
                        if not speech_detected:
                            print("⚠️ لم يتم رصد أي كلام أثناء التسجيل.")
                            return False
                        if audio_data.size == 0:
                            print("⚠️ No audio recorded.")
                            return False
                        write(filename, sample_rate, audio_data)
                        print(f"✅ Recording saved to {filename}")
                        return True

                ring_buffer.append(frame)

    except Exception as e:
        print(f"Error during recording: {e}")
        return False
    finally:
        stream.stop()
        stream.close()


def wait_for_wake_word():
    """Listen for wake word"""
    print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")

    temp_path = TEMP_DIR / f"wake_{int(time.time())}.wav"

    try:
        if record_until_silence(
            filename=str(temp_path), silence_duration=1, max_duration=3
        ):
            result = whisper_model.transcribe(
                str(temp_path),
                language="ar",
                initial_prompt="لبيب هو اسم الروبوت. الكلمات المتوقعة: لبيب",
            )

            text = result["text"].strip().lower()
            print("👂 سمع:", text)

            if any(word in text for word in ["لبيب", "لبي", "لب", "labeeb"]):
                print("✨ تم التعرف على كلمة التنبيه!")
                speak("نعم، كيف اقدر اخدمك؟")
                return True

        return False

    finally:
        if temp_path.exists():
            temp_path.unlink()


def record_and_transcribe(wait_for_wake=True):
    """Record and transcribe speech"""
    global session_active, last_interaction

    # Don't record if session is not active
    if not session_active:
        return ""

    # Don't start recording if Labeeb is speaking
    while is_currently_speaking():
        time.sleep(0.1)

    print("[Labeeb] Listening...")
    temp_path = TEMP_DIR / f"record_{int(time.time())}.wav"

    try:
        print("before record_until_silence")
        if not record_until_silence(str(temp_path)):
            print("❌ لم يتم تسجيل كلام، أعِد المحاولة.")
            return ""

        result = whisper_model.transcribe(
            str(temp_path), language="ar", initial_prompt="توقع كلام باللهجة السعودية"
        )

        text = result["text"].strip()
        print("📄 النص:", text)

        if text == "":
            print("⚠️ النص فارغ بعد التحويل، إعادة التسجيل.")
            return ""

        last_interaction = current_time()
        return text

    except Exception as e:
        print(f"❌ خطأ في التسجيل أو التحويل: {e}")
        return ""

    finally:
        if temp_path.exists():
            temp_path.unlink()


def transcribe_audio_bytes(audio_bytes, sample_rate=16000):
    """Transcribe audio bytes (from Pi) using Whisper."""
    # Save to temp file (Whisper expects a file)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        result = whisper_model.transcribe(
            temp_path, language="ar", initial_prompt="توقع كلام باللهجة السعودية"
        )
        text = result["text"].strip()
        print("📄 النص:", text)
        return text
    finally:
        import os

        if os.path.exists(temp_path):
            os.remove(temp_path)
