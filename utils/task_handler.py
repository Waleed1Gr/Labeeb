from pathlib import Path
import json
from datetime import datetime
from api_clients.embedding_api import get_embedding
from utils.time_parser import parse_date_arabic
from utils.faiss_helper import create_index, add_embedding, search_index  
from api_clients.tts_api import speak
import numpy as _np

# Determine project root and ensure tasks.json is saved/loaded there
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_TASKS_FILE = BASE_DIR / "tasks.json"

# In-memory stores
tasks = []       # Each item: {"text": str, "time": datetime}
embeddings = []  # Each item: list[float]
index = create_index(dimension=384)

def load_tasks():
    """
    Load tasks from tasks.json (if it exists), rebuild the FAISS index,
    and repopulate in-memory `tasks` and `embeddings` lists.

    Any errors during embedding fetch are logged; tasks with failed embeddings
    will still be loaded (but won't be indexed).
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
                    dt = datetime.fromisoformat(entry["time"])
                    text = entry["text"]
                except Exception as parse_err:
                    print(f"⚠️ Skipping invalid task entry: {entry} ({parse_err})")
                    continue

                tasks.append({"text": text, "time": dt})
                try:
                    emb = get_embedding(text)
                    if emb and isinstance(emb, list) and len(emb) == 384:
                        embeddings.append(emb)
                    else:
                        # If embedding is empty or malformed, append a zero vector
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
    except Exception as e:
        print(f"Load tasks error: {e}")

def save_tasks():
    """
    Persist the current `tasks` list to tasks.json in the project root.
    Each entry is saved as {"text": ..., "time": <ISO-8601 string>}.
    """
    try:
        # Ensure parent directory exists (it should, but just in case)
        DATA_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(DATA_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"text": t["text"], "time": t["time"].isoformat()}
                    for t in tasks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"Save tasks error: {e}")

def add_task(text: str):
    """
    Add a new task. Parses any Arabic date in `text`, obtains its embedding,
    appends to in-memory lists, updates FAISS index, persists tasks.json,
    and provides voice feedback.

    If embedding fails, uses a zero vector placeholder to avoid crashing.
    """
    global index
    try:
        dt = parse_date_arabic(text) or datetime.now()

        # Attempt to get embedding
        try:
            emb = get_embedding(text)
            if not emb or not isinstance(emb, list) or len(emb) != 384:
                raise ValueError("Invalid embedding shape")
        except Exception as emb_err:
            print(f"⚠️ Embedding fetch failed for new task '{text}': {emb_err}. Using zero vector.")
            emb = [0.0] * 384

        # Update in-memory stores
        tasks.append({"text": text, "time": dt})
        embeddings.append(emb)

        # Update FAISS index
        try:
            add_embedding(index, emb)
        except Exception as idx_err:
            print(f"⚠️ Failed to add new embedding to FAISS index: {idx_err}")

        # Persist to disk
        save_tasks()

        print(f"✅ سجلت: \"{text}\" @ {dt.isoformat()}")
        speak("تم تسجيل المهمة يا بطل!")
    except Exception as e:
        print(f"Add task error: {e}")
        speak("حصل خطأ في تسجيل المهمة")

def delete_task(query: str):
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
            # Delete the most recently added task (last in the list)
            idx_to_delete = len(tasks) - 1
            task_to_delete = tasks[idx_to_delete]
            print(f"🎯 Detected 'last task' request")
            
        elif any(phrase in query_lower for phrase in ["أول مهمة", "اول مهمة", "المهمة الأولى", "first task"]):
            # Delete the first task
            idx_to_delete = 0
            task_to_delete = tasks[idx_to_delete]
            print(f"🎯 Detected 'first task' request")
            
        else:
            # 🔍 Use semantic similarity search for specific task content
            print(f"🔍 Using semantic search for: '{query}'")
            
            q_emb = get_embedding(query)
            if not q_emb or not isinstance(q_emb, list) or len(q_emb) != 384:
                speak("ما فهمت المهمة اللي تبي تحذفها.")
                return

            indices = search_index(index, q_emb, k=1)
            
            if not indices or indices[0] == -1 or indices[0] >= len(tasks):
                speak("ما لقيت مهمة قريبة من اللي قلتها.")
                return

            idx_to_delete = indices[0]
            task_to_delete = tasks[idx_to_delete]

        # تأكد من أن الفهرس صحيح
        if idx_to_delete is None or idx_to_delete < 0 or idx_to_delete >= len(tasks):
            print(f"❌ Invalid index: {idx_to_delete}, tasks length: {len(tasks)}")
            speak("حصل خطأ في تحديد المهمة.")
            return

        # اعرض المهمة المراد حذفها
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
    If embedding fails or returns invalid, fall back to returning ALL tasks.
    Returns a list of matching tasks (each as {"text": ..., "time": datetime}).
    """
    try:
        if not tasks:
            return []

        # If query contains an Arabic date token, return tasks near that date
        dt = parse_date_arabic(query)
        if dt:
            return [
                t for t in tasks
                if abs((t["time"].date() - dt.date()).days) <= 1
            ]

        # Otherwise, try semantic search
        try:
            q_emb = get_embedding(query)
            if not q_emb or not isinstance(q_emb, list) or len(q_emb) != 384:
                raise ValueError("Invalid query embedding")
            indices = search_index(index, q_emb, k)
            return [tasks[i] for i in indices if i < len(tasks)]
        except Exception as emb_err:
            print(f"⚠️ Embedding fetch failed for search query '{query}': {emb_err}. Falling back to all tasks.")
            # If embedding fails, simply return all tasks
            return tasks.copy()
    except Exception as e:
        print(f"Search tasks error: {e}")
        return []
    
def clear_all_tasks():
    """
    Clears all tasks and embeddings and resets FAISS index.
    """
    global tasks, embeddings, index
    tasks.clear()
    embeddings.clear()
    index = create_index(dimension=384)
    save_tasks()