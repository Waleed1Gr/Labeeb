# listen.py
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import time

fs = 44100
wake_word = "لبيب"
chunk_seconds = 2  # مدة كل تسجيل مؤقت (قصير) بالثواني

def listen_for_wake_word():
    model = whisper.load_model("large")
    print("🎤 جاري الاستماع... قل 'لبيب' لبدء التفاعل.")

    while True:
        # تسجيل مقطع قصير (chunk)
        recording = sd.rec(int(chunk_seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write("chunk.wav", fs, recording)

        # تحويل إلى نص
        result = model.transcribe("chunk.wav", language="ar")
        text = result["text"]
        print("🔎 تم سماع:", text)

        if wake_word in text:
            print("✅ تم سماع كلمة التنبيه!")
            return  # يخرج من الدالة ويكمل البرنامج
        
def record_command():
    print("🟢 مرحباً! ماذا تريد أن أفعل؟ تحدث الآن...")
    seconds = 4  # مدة تسجيل الجملة بعد كلمة لبيب (غيّر الرقم حسب رغبتك)
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("command.wav", fs, recording)
    print("⏹️ انتهى التسجيل.")

    model = whisper.load_model("large")
    result = model.transcribe("command.wav", language="ar")
    print("📝 النص المكتوب:", result["text"])
    return result["text"]
  
    
def main():
    print("Allam LLM audio interface - ready for your instructions.")

if __name__ == "__main__":
    main()
    listen_for_wake_word()
    command_text = record_command()