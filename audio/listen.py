import time
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad
from pydub import AudioSegment
import tempfile
from utils.config import SESSION_TIMEOUT, last_interaction, session_active, TEMP_DIR
from utils.utils import current_time
from audio.speak import speak, stop_current_speech, is_currently_speaking
from models.model import whisper_model

# Initialize VAD
vad = webrtcvad.Vad(1)  # Sensitivity level 2 (0-3)

# Adjust silence detection parameters
SILENCE_THRESHOLD = 15  # Reduced from 20
MIN_AUDIO_LEVEL = 0.003  # Minimum audio level to consider as speech

def is_speech(frame, sample_rate=16000):
    """Check if audio frame contains speech"""
    try:
        return vad.is_speech(frame.tobytes(), sample_rate)
    except:
        return False

def wait_for_wake_word():
    """Listen for wake word with improved silence detection"""
    print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
    
    # Ensure temp directory exists
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_samples = int(sample_rate * frame_duration / 1000)
    silence_threshold = 20
    silent_chunks = 0
    chunks = []

    def callback(indata, frames, time, status):
        nonlocal silent_chunks
        audio = np.frombuffer(indata, dtype=np.int16)
        if is_speech(audio):
            silent_chunks = 0
            chunks.append(audio.copy())
        else:
            silent_chunks += 1
        if silent_chunks > silence_threshold:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                          callback=callback, blocksize=frame_samples):
            # Reduced sleep time for better responsiveness
            sd.sleep(2000)  # 2 seconds instead of 3
    except sd.CallbackStop:
        pass

    if not chunks:  # No speech detected
        print("🔇 لا يوجد صوت كافي، إعادة المحاولة...")
        return False

    # Use Path for proper file handling
    temp_path = TEMP_DIR / f"wake_{int(time.time())}.wav"
    
    try:
        AudioSegment(
            np.concatenate(chunks).tobytes(), 
            frame_rate=sample_rate, 
            sample_width=2, 
            channels=1
        ).export(str(temp_path), format="wav")

        result = whisper_model.transcribe(
            str(temp_path),
            language="ar",
            initial_prompt="لبيب هو اسم الروبوت. الكلمات المتوقعة: لبيب"
        )

        text = result["text"].strip().lower()
        print("👂 سمع:", text)

        if any(word in text for word in ["لبيب", "لبي", "لب", "labeeb"]):
            print("✨ تم التعرف على كلمة التنبيه!")
            speak("نعم، كيف اقدر اخدمك؟")
            return True

        return False

    except Exception as e:
        print(f"❌ خطأ في التسجيل: {e}")
        return False

    finally:
        if temp_path.exists():
            temp_path.unlink()

def record_and_transcribe(wait_for_wake=True):
    """Record and transcribe speech with improved silence detection"""
    global session_active, last_interaction
    
    # Small delay before recording to prevent false triggers
    time.sleep(0.2)
    
    if wait_for_wake and not session_active:
        if not wait_for_wake_word():
            return ""
        session_active = True
        last_interaction = current_time()
    
    print("[Labeeb] Listening...")
    sample_rate = 16000
    frame_duration = 30
    frame_samples = int(sample_rate * frame_duration / 1000)
    silence_threshold = 20
    silent_chunks = 0
    chunks = []

    def callback(indata, frames, time, status):
        nonlocal silent_chunks
        audio = np.frombuffer(indata, dtype=np.int16)
        
        # Check both VAD and amplitude
        is_speech_detected = is_speech(audio) or np.abs(audio).mean() > MIN_AUDIO_LEVEL
        
        if is_speech_detected:
            silent_chunks = 0
            chunks.append(audio.copy())
        else:
            silent_chunks += 1
        if silent_chunks > SILENCE_THRESHOLD:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                          blocksize=frame_samples, callback=callback):
            sd.sleep(10000)  # Max 10 seconds
    except sd.CallbackStop:
        pass

    if not chunks:
        print("❌ لم يتم تسجيل كلام، أعِد المحاولة.")
        return ""

    audio = np.concatenate(chunks)
    temp_path = TEMP_DIR / f"record_{int(time.time())}.wav"
    
    try:
        AudioSegment(
            audio.tobytes(), 
            frame_rate=sample_rate, 
            sample_width=2, 
            channels=1
        ).export(str(temp_path), format="wav")

        result = whisper_model.transcribe(
            str(temp_path),
            language="ar",
            initial_prompt="توقع كلام باللهجة السعودية"
        )

        text = result["text"].strip()
        print("📄 النص:", text)

        if text == "":
            print("⚠️ النص فارغ بعد التحويل، إعادة التسجيل.")
            return ""

        # Update interaction time if valid input received
        last_interaction = current_time()
    
        if session_active and (current_time() - last_interaction > SESSION_TIMEOUT):
            session_active = False
            print("\n💤 انتهت الجلسة")
            speak("تم إنهاء الجلسة، ناديني إذا احتجتني!")
            
        return text

    except Exception as e:
        print(f"❌ خطأ في التسجيل أو التحويل: {e}")
        return ""

    finally:
        if temp_path.exists():
            temp_path.unlink()
