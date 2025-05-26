### File: app.py
from tasks.task_manager import add_task, find_related_tasks

# Example usage
if __name__ == "__main__":
    print("1. Adding a task...")
    task = add_task("Remind me to water the plants tomorrow at 6pm", "2025-05-26T18:00:00")
    print("Added Task:", task)

    print("\n2. Querying a related task...")
    results = find_related_tasks("What do I need to do tomorrow?")
    for r in results:
        print(f"- {r['text']} (Scheduled at {r['metadata']['time']})")
