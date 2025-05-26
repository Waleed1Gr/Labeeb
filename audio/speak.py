import pyttsx3
import sys

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        speak(' '.join(sys.argv[1:]))
    else:
        text = input("Enter text to speak: ")
        speak(text)