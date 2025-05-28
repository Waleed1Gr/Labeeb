# listen.py
import torch
import whisper
import sounddevice as sd
from scipy.io.wavfile import write

def lisning():
    fs = 44100
    seconds = 5
    print("🎙️ تسجيل...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("recorded.wav", fs, recording)
    print("✅ تم الحفظ")

    model = whisper.load_model("large")
    result = model.transcribe("recorded.wav", language="ar")
    print("📄 النص:", result["text"])


    string = result["text"]
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(string)
    return string
    
    
def main():
    print("Allam LLM audio interface - ready for your instructions.")

if __name__ == "__main__":
    main()
    print(lisning())