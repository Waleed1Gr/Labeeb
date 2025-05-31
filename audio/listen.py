import time
import os
import queue
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import webrtcvad
from utils.config import SESSION_TIMEOUT, last_interaction, session_active
from utils.utils import current_time
from audio.speak import speak
from models.model import whisper_model

# ============== WAKE WORD & RECORD ==============
def wait_for_wake_word():
    fs = 16000  # Sample rate
    duration = 3  # Increased duration for better capture
    print("🔊 بانتظار كلمة التنبيه: 'لبيب'...")
    
    while True:
        try:
            # Record audio
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            
            amplitude = np.abs(recording).mean()
            print("متوسط الصوت:", amplitude)

            write("wake_check.wav", fs, recording)

            # Transcribe with more context and guidance
            result = whisper_model.transcribe(
                "wake_check.wav",
                language="ar",
                initial_prompt="لبيب هو اسم الروبوت. الكلمات المتوقعة: لبيب"
            )
            
            text = result["text"].strip().lower()
            if(amplitude < 0.002):
                print("🔇 لا يوجد صوت كافي، إعادة المحاولة...")
            else:
                print("👂 سمع:", text)

            # More flexible matching
            if any(word in text for word in ["لبيب", "لبي", "لب", "labeeb"]):
                print("✨ تم التعرف على كلمة التنبيه!")
                speak("نعم، كيف اقدر اخدمك؟")
                return True
            
            time.sleep(0.1)  # Small delay to prevent CPU overload
            
        except Exception as e:
            print(f"❌ خطأ في التعرف على الصوت: {e}")
            time.sleep(1)
            continue



# record_until_silence ############
def record_until_silence_fixed(filename, sample_rate=16000, frame_duration_ms=30, silence_duration=2, max_duration=40):
    import collections

    vad = webrtcvad.Vad(2)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback)
    stream.start()

    frames = []
    ring_buffer = collections.deque()
    silence_start = None
    start_time = time.time()

    frame_size = int(sample_rate * frame_duration_ms / 1000)  # عدد العينات لكل إطار
    bytes_per_frame = frame_size * 2  # 16-bit = 2 bytes per sample

    buffer = b""
    speech_detected = False  # اكتشاف إذا تم الكشف عن الكلام

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
                    speech_detected = True  # <-- إذا تم اكتشاف كلام، فعل العلم
                else:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("🔇 Silence detected, stopping recording.")
                        stream.stop()
                        stream.close()
                        audio_data = np.frombuffer(b''.join(ring_buffer), dtype=np.int16)
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
        stream.stop()
        stream.close()
        return False

    # Timeout or max duration reached
    stream.stop()
    stream.close()
    audio_data = np.frombuffer(b''.join(ring_buffer), dtype=np.int16)
    if audio_data.size == 0:
        print("⚠️ No audio recorded.")
        return False
    write(filename, sample_rate, audio_data)
    print(f"✅ Recording saved to {filename}")
    return True

def record_and_transcribe(wait_for_wake=True):
    global session_active, last_interaction
    
    if wait_for_wake and not session_active:
        if not wait_for_wake_word():
            return ""
        session_active = True
        last_interaction = current_time()
    
    filename = f"record_{int(time.time())}.wav"
    try:
        print('before record_until_silence')
        success = record_until_silence_fixed(filename, sample_rate=16000, silence_duration=2, max_duration=10)
        print(success)
        print(type(success))
        if not success:
            print("❌ لم يتم تسجيل كلام، أعِد المحاولة.")
            return ""

        result = whisper_model.transcribe(
            filename,
            language="ar",
            initial_prompt="توقع كلام باللهجة السعودية"
        )
        
        try:
            os.remove(filename)
        except:
            pass
            
        text = result["text"].strip()
        print("📄 النص:", text)

        if text == "":
            print("⚠️ النص فارغ بعد التحويل، إعادة التسجيل.")
            return ""

        if session_active:
            last_interaction = current_time()
            
        if session_active and (current_time() - last_interaction > SESSION_TIMEOUT):
            session_active = False
            print("\n💤 انتهت الجلسة")
            speak("تم إنهاء الجلسة، ناديني إذا احتجتني!")
            
        return text
        
    except Exception as e:
        print(f"❌ خطأ في التسجيل أو التحويل: {e}")
        return ""
