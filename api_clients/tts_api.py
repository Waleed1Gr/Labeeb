# api_clients/tts_api.py

from dotenv import load_dotenv
load_dotenv()

import os
import time
from pathlib import Path
from openai import OpenAI, OpenAIError
import simpleaudio as sa  # ✅ REPLACEMENT: use simpleaudio

# Initialize OpenAI client for TTS
_tts_api_key = os.getenv("OPENAI_API_KEY")
_tts_client = None
try:
    _tts_client = OpenAI(api_key=_tts_api_key)
except OpenAIError as e:
    print(f"Error initializing TTS OpenAI client: {e}")

# Filenames
_RESPONSE_FILENAME = Path("response.wav")
_WARNING_FILENAME = Path("warning.wav")
_TEMP_RESPONSE = Path("response_tmp.wav")
_TEMP_WARNING = Path("warning_tmp.wav")


def _play_wav(path: Path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(str(path))
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"simpleaudio playback error: {e}")


def speak(text: str,
          voice: str = "echo",
          instructions: str = "تكلم بلهجة سعودية نجدية واضحة، لا تغلط بالكلمات وخلك ذكي"):
    try:
        text = text.replace("<close_conversation>", "").strip()

        if _TEMP_RESPONSE.exists():
            try:
                _TEMP_RESPONSE.unlink()
            except:
                pass

        try:
            with _tts_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="echo",
                input=text,
                response_format="wav",
                instructions=instructions
            ) as response:
                response.stream_to_file(_TEMP_RESPONSE)
        except Exception as stream_err:
            print(f"Speech streaming error: {stream_err}")
            return

        # ✅ PLAY .wav using simpleaudio
        if _TEMP_RESPONSE.exists():
            _play_wav(_TEMP_RESPONSE)

        # Replace response.wav after playback
        if _TEMP_RESPONSE.exists():
            try:
                if _RESPONSE_FILENAME.exists():
                    _RESPONSE_FILENAME.unlink()
            except:
                pass
            try:
                _TEMP_RESPONSE.replace(_RESPONSE_FILENAME)
            except Exception as rename_err:
                print(f"Error renaming temp response file: {rename_err}")

    except Exception as e:
        print(f"Speech error: {e}")


def speak_warning(text: str,
                  voice: str = "ash",
                  instructions: str = "تكلم بلهجة سعودية واضحة، لا تغلط بالكلمات وخلك ذكي"):
    try:
        if _TEMP_WARNING.exists():
            try:
                _TEMP_WARNING.unlink()
            except:
                pass

        try:
            with _tts_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="echo",
                input=text,
                response_format="wav",
                instructions=instructions
            ) as response:
                response.stream_to_file(_TEMP_WARNING)
        except Exception as stream_err:
            print(f"Warning streaming error: {stream_err}")
            return

        # ✅ PLAY .wav using simpleaudio
        if _TEMP_WARNING.exists():
            _play_wav(_TEMP_WARNING)

        # Replace warning.wav after playback
        if _TEMP_WARNING.exists():
            try:
                if _WARNING_FILENAME.exists():
                    _WARNING_FILENAME.unlink()
            except:
                pass
            try:
                _TEMP_WARNING.replace(_WARNING_FILENAME)
            except Exception as rename_err:
                print(f"Error renaming temp warning file: {rename_err}")

    except Exception as e:
        print(f"Warning speech error: {e}")
