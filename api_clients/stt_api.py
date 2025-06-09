from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from openai import OpenAI, OpenAIError

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in .env or environment)
api_key = os.getenv("OPENAI_API_KEY")
client = None
try:
    client = OpenAI(api_key=api_key)
except OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")

def transcribe_from_file(filepath: Path, language: str = "ar", prompt: str = "توقع كلام باللهجة السعودية") -> str:
    """
    Transcribes the given WAV/MP3 file using OpenAI Whisper API.
    Returns the transcript as a lowercase string, or an empty string on failure.
    """
    try:
        with open(filepath, "rb") as audio_file:
            # Call the Whisper endpoint; `response` is a Transcription object
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json",
                language=language,
                prompt=prompt
            )
        # The Transcription object exposes the text via .text
        transcript = getattr(response, "text", None)
        if transcript is None:
            # In some SDK versions, it might be under `.text` or `.text` inside choices
            # But in newer OpenAI Python SDK, response.text is correct.
            # If no .text attribute, try dict‐style fallback:
            try:
                return response["text"].strip().lower()
            except Exception:
                return ""
        return transcript.strip().lower()
    except Exception as e:
        print(f"STT API error: {e}")
        return ""
