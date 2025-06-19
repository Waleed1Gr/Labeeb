# Updated task_handler.py with smart time handling

from pathlib import Path
import json
from datetime import datetime
from api_clients.embedding_api import get_embedding
from utils.time_parser import parse_date_arabic
from utils.faiss_helper import create_index, add_embedding, search_index  
from api_clients.tts_api import speak
import numpy as _np

# ✨ Import the new task extractor
from utils.task_extractor import extract_task_content

# Determine project root and ensure tasks.json is saved/loaded there
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_TASKS_FILE = BASE_DIR / "tasks.json"

# In-memory stores
tasks = []       # Each item: {"text": str, "time": datetime, "time_specified": bool}
embeddings = []  # Each item: list[float]
index = create_index(dimension=384)

def load_tasks():
    """
    Load tasks from tasks.json (if it exists), rebuild the FAISS index,
    and repopulate in-memory `tasks` and `embeddings` lists.
    """
    global tasks, embeddings, index
    try:
        if DATA_TASKS_FILE.exists():
            with open(DATA_TASKS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            tasks.clear()
            embeddings.clear()
            index = create_index(dimension=384)

            for entry in loaded:
                try:
                    text = entry["text"]
                    time_str = entry["time"]
                    # Convert time string to datetime for backward compatibility
                    if isinstance(time_str, str):
                        dt = datetime.fromisoformat(time_str)
                    else:
                        dt = time_str
                        
                except Exception as parse_err:
                    print(f"⚠️ Skipping invalid task entry: {entry} ({parse_err})")
                    continue

                tasks.append({
                    "text": text, 
                    "time": dt
                })
                
                try:
                    emb = get_embedding(text)
                    if emb and isinstance(emb, list) and len(emb) == 384:
                        embeddings.append(emb)
                    else:
                        print(f"⚠️ Received invalid embedding for task '{text}', using zero vector instead.")
                        embeddings.append([0.0] * 384)
                except Exception as emb_err:
                    print(f"⚠️ Embedding fetch failed for '{text}': {emb_err}.\n   Using zero vector instead.")
                    embeddings.append([0.0] * 384)

            # Only add to FAISS if we have at least one embedding
            if embeddings:
                try:
                    index.add(_np.array(embeddings, dtype=_np.float32))
                except Exception as idx_err:
                    print(f"⚠️ Failed to add embeddings to FAISS index: {idx_err}")
                    
            print(f"✅ Loaded {len(tasks)} tasks")
            
            # 🔧 Auto-save in new format if we fixed any old tasks
            needs_save = any("time_specified" not in entry for entry in loaded if isinstance(entry, dict))
            if needs_save:
                print("🔧 Auto-updating tasks.json to new format...")
                save_tasks()  # This will save with the new format
    except Exception as e:
        print(f"Load tasks error: {e}")

def save_tasks():
    """
    Persist the current `tasks` list to tasks.json in the project root.
    Each entry is saved as {"text": ..., "time": <ISO-8601 string>, "time_specified": bool}.
    """
    try:
        DATA_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(DATA_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "text": t["text"], 
                        "time": t["time"].isoformat() if isinstance(t["time"], datetime) else t["time"]
                    }
                    for t in tasks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"Save tasks error: {e}")

def add_task(raw_input: str, conversation_history=None):
    """
    Add a new task with improved content extraction and smart time handling.
    Now with context awareness for handling references to previous conversations.
    
    Args:
        raw_input: The full user input (e.g., "سجل مهمة أداء الاختبار بكرة")
        conversation_history: List of recent conversation exchanges
    """
    global index
    try:
        # ✨ Extract clean task content with context awareness
        if conversation_history:
            from utils.task_extractor import extract_task_with_context
            clean_task = extract_task_with_context(raw_input, conversation_history)
        else:
            from utils.task_extractor import extract_task_content
            clean_task = extract_task_content(raw_input)

        print(f"📝 Raw input: '{raw_input}'")
        print(f"✨ Extracted task: '{clean_task}'")
        print(f"📅 Recorded at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Attempt to get embedding for the clean task content
        try:
            emb = get_embedding(clean_task)
            if not emb or not isinstance(emb, list) or len(emb) != 384:
                raise ValueError("Invalid embedding shape")
        except Exception as emb_err:
            print(f"⚠️ Embedding fetch failed for new task '{clean_task}': {emb_err}. Using zero vector.")
            emb = [0.0] * 384

        # Update in-memory stores with clean task content
        tasks.append({
            "text": clean_task, 
            "time": datetime.now().isoformat()  # ✨ Just when you recorded it
        })
        embeddings.append(emb)

        # Update FAISS index
        try:
            add_embedding(index, emb)
        except Exception as idx_err:
            print(f"⚠️ Failed to add new embedding to FAISS index: {idx_err}")

        # Persist to disk
        save_tasks()

        print(f"✅ سجلت: \"{clean_task}\" @ {datetime.now().isoformat()}")
        speak("تم تسجيل المهمة يا بطل!")
    except Exception as e:
        print(f"Add task error: {e}")
        speak("حصل خطأ في تسجيل المهمة")

def delete_task(query: str):
    """
    Delete a task with improved query processing.
    """
    global tasks, embeddings, index

    try:
        print(f"🔍 Delete request: '{query}'")
        print(f"📋 Current tasks count: {len(tasks)}")
        
        if not tasks:
            speak("ما عندك مهام عشان أحذفها.")
            return

        query_lower = query.lower().strip()
        idx_to_delete = None
        task_to_delete = None

        # 🎯 Handle specific deletion patterns
        if any(phrase in query_lower for phrase in ["آخر مهمة", "اخر مهمة", "المهمة الأخيرة", "last task"]):
            idx_to_delete = len(tasks) - 1
            task_to_delete = tasks[idx_to_delete]
            print(f"🎯 Detected 'last task' request")
            
        elif any(phrase in query_lower for phrase in ["أول مهمة", "اول مهمة", "المهمة الأولى", "first task"]):
            idx_to_delete = 0
            task_to_delete = tasks[idx_to_delete]
            print(f"🎯 Detected 'first task' request")
            
        else:
            # 🔍 Use semantic similarity search for specific task content
            from utils.task_extractor import extract_delete_query
            clean_query = extract_delete_query(query)
            print(f"🔍 Using semantic search for: '{clean_query}'")
            
            q_emb = get_embedding(clean_query)
            if not q_emb or not isinstance(q_emb, list) or len(q_emb) != 384:
                speak("ما فهمت المهمة اللي تبي تحذفها.")
                return

            indices = search_index(index, q_emb, k=1)
            
            if not indices or indices[0] == -1 or indices[0] >= len(tasks):
                speak("ما لقيت مهمة قريبة من اللي قلتها.")
                return

            idx_to_delete = indices[0]
            task_to_delete = tasks[idx_to_delete]

        if idx_to_delete is None or idx_to_delete < 0 or idx_to_delete >= len(tasks):
            print(f"❌ Invalid index: {idx_to_delete}, tasks length: {len(tasks)}")
            speak("حصل خطأ في تحديد المهمة.")
            return

        print(f"🗑️ Deleting task: {task_to_delete['text']} (Index: {idx_to_delete})")
        
        # نحذف المهمة والembedding
        deleted_task = tasks.pop(idx_to_delete)
        embeddings.pop(idx_to_delete)

        # نعيد بناء الفهرس بالكامل
        index = create_index(dimension=384)
        if embeddings:
            index.add(_np.array(embeddings, dtype=_np.float32))

        save_tasks()

        print(f"✅ تم حذف المهمة: {deleted_task['text']}")
        speak(f"تمام، حذفت لك مهمة: {deleted_task['text']}.")
        
    except Exception as e:
        print(f"❌ Delete task error: {e}")
        import traceback
        traceback.print_exc()
        speak("حصل خطأ في حذف المهمة.")

def search_tasks(query: str, k: int = 5) -> list:
    """
    Search tasks by date if a date keyword is present (±1 day),
    otherwise perform a k-NN search in the FAISS index for semantic similarity.
    Returns a list of matching tasks in their original format.
    """
    try:
        if not tasks:
            return []

        # If query contains an Arabic date token, return tasks near that date
        dt_result, _ = parse_date_arabic(query)
        if dt_result:
            matching_tasks = [
                t for t in tasks
                if abs((t["time"].date() - dt_result.date()).days) <= 1
            ]
            print(f"🔍 Date search found {len(matching_tasks)} tasks")
            return matching_tasks

        # Otherwise, try semantic search
        try:
            q_emb = get_embedding(query)
            if not q_emb or not isinstance(q_emb, list) or len(q_emb) != 384:
                raise ValueError("Invalid query embedding")
            indices = search_index(index, q_emb, k)
            matching_tasks = [tasks[i] for i in indices if i < len(tasks)]
            print(f"🔍 Semantic search found {len(matching_tasks)} tasks")
            return matching_tasks
        except Exception as emb_err:
            print(f"⚠️ Embedding fetch failed for search query '{query}': {emb_err}. Falling back to all tasks.")
            # Return all tasks in original format for fallback
            print(f"🔍 Fallback returning {len(tasks)} tasks")
            return tasks.copy()
    except Exception as e:
        print(f"Search tasks error: {e}")
        return []

def format_task_for_display(task: dict) -> dict:
    """
    Format a task for display, showing time only if it was explicitly specified.
    """
    display_task = task.copy()
    
    # If time was not explicitly specified, we show just the task text
    # The internal timestamp is kept for sorting/filtering but not shown to user
    if not task["time_specified"]:
        # Keep the original clean task text without showing the internal timestamp
        pass  # display_task["text"] already contains the clean text
    
    return display_task

def get_all_tasks_for_display() -> list:
    """
    Get all tasks for display - just return the task texts from JSON.
    """
    return [task["text"] for task in tasks]

def find_task_recording_time(query: str):
    """Find when a specific task was recorded."""
    try:
        query_lower = query.lower()
        
        # Extract task keywords from query like "متى سجلت مهمة الاختبار؟"
        import re
        clean_query = re.sub(r'متى\s+سجلت\s+(مهمة\s+)?', '', query_lower).strip()
        clean_query = re.sub(r'[؟?]', '', clean_query).strip()
        
        if not clean_query:
            speak("أي مهمة تقصد؟")
            return
        
        # Search for matching task
        for task in tasks:
            if any(word in task["text"].lower() for word in clean_query.split()):
                # Format the recorded time nicely
                recorded_dt = datetime.fromisoformat(task["time"])
                formatted_time = recorded_dt.strftime("%d %B الساعة %H:%M")
                
                speak(f"سجلت مهمة '{task['text']}' يوم {formatted_time}")
                return
        
        speak("ما لقيت المهمة اللي تقصدها")
        
    except Exception as e:
        print(f"❌ Error finding task recording time: {e}")
        speak("حصل خطأ في البحث عن المهمة")

def find_task_recording_time(query: str):
    """Find when a specific task was recorded."""
    try:
        query_lower = query.lower()
        
        # Extract task keywords from query like "متى سجلت مهمة الاختبار؟"
        import re
        clean_query = re.sub(r'متى\s+سجلت\s+(مهمة\s+)?', '', query_lower).strip()
        clean_query = re.sub(r'[؟?]', '', clean_query).strip()
        
        if not clean_query:
            speak("أي مهمة تقصد؟")
            return
        
        # Search for matching task
        for task in tasks:
            if any(word in task["text"].lower() for word in clean_query.split()):
                # Use the time field (when you recorded it)
                if isinstance(task["time"], str):
                    recorded_dt = datetime.fromisoformat(task["time"])
                else:
                    recorded_dt = task["time"]
                    
                formatted_time = recorded_dt.strftime("%d %B الساعة %H:%M")
                speak(f"سجلت مهمة '{task['text']}' يوم {formatted_time}")
                return
        
        speak("ما لقيت المهمة اللي تقصدها")
        
    except Exception as e:
        print(f"❌ Error finding task recording time: {e}")
        speak("حصل خطأ في البحث عن المهمة")

def clear_all_tasks():
    """
    Clears all tasks and embeddings and resets FAISS index.
    """
    global tasks, embeddings, index
    tasks.clear()
    embeddings.clear()
    index = create_index(dimension=384)
    save_tasks()

def extract_delete_query(query: str) -> str:
    """Extract meaningful content from delete queries."""
    import re
    
    delete_patterns = [
        r'^احذف\s+مهمة\s+',
        r'^احذف\s+لي\s+مهمة\s+',
        r'^احذف\s+',
        r'^شيل\s+مهمة\s+',
        r'^شيل\s+',
        r'^امسح\s+مهمة\s+',
        r'^امسح\s+',
    ]
    
    cleaned = query.strip()
    for pattern in delete_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        if cleaned != query:
            break
    
    return cleaned if cleaned else query