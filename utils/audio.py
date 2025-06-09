import webrtcvad
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

def record_until_silence_fixed(
    filename: str,
    sample_rate: int = 16000,
    frame_duration_ms: int = 30,
    silence_duration: int = 3,
    max_duration: int = 60
) -> bool:
    """
    Records audio from the microphone until a period of silence is detected,
    then writes it to `filename`. Returns True on success, False otherwise.
    """
    import collections

    vad = webrtcvad.Vad(2)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        callback=callback
    )
    stream.start()

    ring_buffer = collections.deque()
    silence_start = None
    start_time = time.time()
    buffer = b""

    frame_size = int(sample_rate * frame_duration_ms / 1000)
    bytes_per_frame = frame_size * 2  # 16-bit = 2 bytes per sample
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
                    speech_detected = True  # اكتشاف إذا تم الكشف عن الكلام
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

    # If we exit the loop (max duration reached), still save whatever we have:
    stream.stop()
    stream.close()
    audio_data = np.frombuffer(b''.join(ring_buffer), dtype=np.int16)
    if audio_data.size == 0:
        print("⚠️ No audio recorded.")
        return False
    write(filename, sample_rate, audio_data)
    print(f"✅ Recording saved to {filename}")
    return True
