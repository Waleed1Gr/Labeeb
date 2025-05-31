from pathlib import Path
import time
import os
from pydub import AudioSegment
from utils.config import client

# ============== SPEAK FUNCTION ==============
def speak(text):
    try:
        # Remove <close_conversation> tag before speaking
        text = text.replace("<close_conversation>", "").strip()
        
        filename = f"response_{int(time.time())}.mp3"
        path = Path(filename)

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="verse",
            input=text,
            instructions="تكلم بلهجة سعودية نجدية واضحة، لا تغلط بالكلمات وخلك ذكي"
        ) as response:
            response.stream_to_file(path)

        audio = AudioSegment.from_file(path)
        duration_sec = len(audio) / 1000

        if os.name == "nt":
            os.system(f'start /min /wait "" "{path}"')
        else:
            os.system(f"afplay {path}")

        time.sleep(duration_sec)
        
        # Clean up the file
        if path.exists():
            os.remove(path)
            
    except Exception as e:
        print(f"Speech error: {e}")
