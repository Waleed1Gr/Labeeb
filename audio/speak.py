from pathlib import Path
import time
import threading
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import io
from utils.config import client, TEMP_DIR

# Global variables for speech control
playback_thread = None
current_play_obj = None
stop_speaking_flag = False
is_speaking = False
speech_lock = threading.Lock()

def play_audio(audio_data):
    """Play audio data in a separate thread"""
    global current_play_obj, is_speaking
    try:
        with speech_lock:
            is_speaking = True
            audio = AudioSegment.from_file(audio_data, format="wav")
            play_obj = _play_with_simpleaudio(audio)
            current_play_obj = play_obj
            
            while play_obj.is_playing() and not stop_speaking_flag:
                time.sleep(0.1)
            
            if stop_speaking_flag:
                play_obj.stop()
                print("[Labeeb] Speech interrupted.")
            
    finally:
        with speech_lock:
            is_speaking = False
            current_play_obj = None

def speak(text):
    """Generate and play speech for given text"""
    global playback_thread, stop_speaking_flag
    stop_speaking_flag = False
    
    try:
        text_clean = text.replace("<close_conversation>", "").strip()
        
        # Generate audio
        audio_data = io.BytesIO()
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="verse",
            input=text_clean,
            response_format="wav",
            instructions="تكلم بلهجة سعودية نجدية واضحة"
        ) as response:
            for chunk in response.iter_bytes():
                if stop_speaking_flag:
                    return
                audio_data.write(chunk)
        
        if not stop_speaking_flag:
            audio_data.seek(0)
            playback_thread = threading.Thread(target=play_audio, args=(audio_data,))
            playback_thread.start()
            
    except Exception as e:
        print(f"Speech error: {e}")

def is_currently_speaking():
    """Thread-safe check if speech is ongoing"""
    with speech_lock:
        return is_speaking

def stop_current_speech():
    """Thread-safe speech interruption"""
    global stop_speaking_flag
    with speech_lock:
        if current_play_obj and current_play_obj.is_playing():
            stop_speaking_flag = True
            current_play_obj.stop()
