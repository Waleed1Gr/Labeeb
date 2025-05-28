### File: main.py
from tasks.task_manager import add_task, find_related_tasks

# Example usage
if __name__ == "__main__":
    print("1. Adding a task...")
    task = add_task(
        "Remind me to water the plants tomorrow at 6pm", "2025-05-26T18:00:00"
    )
    print("Added Task:", task)

    print("\n2. Querying a related task...")
    results = find_related_tasks("What do I need to do tomorrow?")
    for r in results:
        print(f"- {r['text']} (Scheduled at {r['metadata']['time']})")

# this is the senario of labeeb:
# --------------------------------------------------------------------------------------------------------------------------
# 1- camera runs to get the image of the user, if user is human and 5 seconds passed, then it will greet the user:

# 2- labeeb will start recording the audio of the user for 6 seconds.

# 3- labeeb will get the audio and turn it into text using whisper. then send it to the LLM to get the response.

# 4- labeeb will get the response from the LLM and send it to TTS to speak it out loud.
