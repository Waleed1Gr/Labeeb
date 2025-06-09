from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

# ============== CONFIGURATION ==============
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("OPENAI_API_KEY")  #just fixed this line to use the traditional way of getting the API key
client = OpenAI(api_key=API_KEY)

speech_file_path = Path("response.mp3")
speech_file_pathp = Path("phone_warning.mp3")
DATA_TASKS_FILE = Path("tasks.json")
wake_word = "لبيب"
SESSION_TIMEOUT = 60  # seconds
session_active = False
last_interaction = 0
