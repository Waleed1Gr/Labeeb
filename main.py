### File: main.py
from tasks.task_manager import add_task, find_related_tasks

# Example usage
if __name__ == "__main__":

    # this is the senario of labeeb:
    # --------------------------------------------------------------------------------------------------------------------------

    # 1- camera runs to get the image of the user, if user is human and 5 seconds passed, then it will greet the user:

    from vision.camera import is_attended
    from audio.speak import speak

    if is_attended(duration=5):
        speak("أهلا، انا لبيب، كيف يمكنني مساعدتك اليوم؟")
    # --------------------------------------------------------------------------------------------------------------------------

    # 2- labeeb will start recording the audio of the user for 6 seconds.

    from audio.listen import record_audio

    audio_file = record_audio(duration=6)

    # --------------------------------------------------------------------------------------------------------------------------

    # 3- labeeb will get the audio and turn it into text using whisper. then send it to the LLM to get the response.

    from audio.listen import transcribe_audio
    from agents.LLM import get_response

    text = transcribe_audio(audio_file)
    # --------------------------------------------------------------------------------------------------------------------------

    # 4- labeeb will get the response from the LLM and send it to TTS to speak it out loud.

    response = get_response(text)
    speak(response)

# --------------------------------------------------------------------------------------------------------------------------

# TODO: try to find a way to send the prompt to the RAG system if it was a task related prompt, and then get the response from the RAG system.
