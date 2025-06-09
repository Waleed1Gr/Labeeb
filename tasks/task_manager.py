from datetime import datetime, timedelta
import json
import re
import faiss
import numpy as np
from utils.config import client, DATA_TASKS_FILE
from audio.speak import speak
from models.model import sentence_model

# Global variables
tasks = []
embeddings = []
index = None  # Will be initialized when needed

def initialize_index():
    """Initialize the FAISS index"""
    global index
    if index is None:
        dimension = 384  # SentenceTransformer embedding dimension
        index = faiss.IndexFlatL2(dimension)

# ============== TASK SYSTEM ==============
def parse_date_arabic(text):
    text = text.strip().lower()
    now = datetime.now()
    if "اليوم" in text:
        return now
    elif "بكرة" in text:
        return now + timedelta(days=1)
    elif "بعد بكرة" in text:
        return now + timedelta(days=2)
    elif "نهاية الأسبوع" in text:
        return now + timedelta(days=(4 - now.weekday()) % 7)
    elif match := re.search(r"(ال)?(جمعة|سبت|أحد|اثنين|ثلاثاء|اربعاء|خميس)( الجاية)?", text):
        weekdays = {"سبت": 5, "أحد": 6, "اثنين": 0, "ثلاثاء": 1, "اربعاء": 2, "خميس": 3, "جمعة": 4}
        delta = (weekdays[match.group(2)] - now.weekday()) % 7 or 7
        return now + timedelta(days=delta)
    elif match := re.search(r"(\d{1,2}) (يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر)", text):
        months = {"يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5,
                  "يونيو": 6, "يوليو": 7, "أغسطس": 8, "سبتمبر": 9,
                  "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12}
        day, month = int(match.group(1)), months[match.group(2)]
        year = now.year + ((datetime(now.year, month, day) < now) and 1)
        return datetime(year, month, day)
    elif match := re.search(r"الساعة (\d{1,2})([:٫،](\d{1,2}))?", text):
        hour, minute = int(match.group(1)), int(match.group(3) or 0)
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return None

def save_tasks():
    try:
        with open(DATA_TASKS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"text": t["text"], "time": t["time"].isoformat()} for t in tasks], f, ensure_ascii=False)
    except Exception as e:
        print(f"Save tasks error: {e}")

def load_tasks():
    global tasks, embeddings, index
    initialize_index()
    try:
        if DATA_TASKS_FILE.exists():
            with open(DATA_TASKS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            tasks.clear()
            embeddings.clear()
            index.reset()
            for t in loaded:
                dt = datetime.fromisoformat(t["time"])
                tasks.append({"text": t["text"], "time": dt})
                emb = sentence_model.encode([t["text"]])[0]
                embeddings.append(emb)
            if embeddings:
                index.add(np.array(embeddings))
    except Exception as e:
        print(f"Load tasks error: {e}")

def classify_input(user_input):
    try:
        prompt = f"""أنت مساعد شخصي ذكي باللهجة السعودية.
        المستخدم قال: "{user_input}"
        
        حلل قصد المستخدم واختر من التالي:
        1. إذا يقصد تسجيل موعد أو مهمة جديدة، رد: تسجيل
        2. إذا يستفسر عن المهام أو المواعيد المسجلة، رد: تذكير
        3. إذا يريد حذف مهمة، رد: حذف
        4. إذا يريد محادثة عادية، رد: محادثة

        رد بخيار واحد من الخيارات المذكورة أعلاه.
        """
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Classify input error: {e}")
        return "محادثة"

def extract_task_summary(text):
    """Extract a concise task summary from user input"""
    try:
        prompt = f"""المستخدم قال باللهجة السعودية: "{text}"

        استخرج عنوان مختصر للمهمة من كلام المستخدم. العنوان يجب أن:
        1. يكون مختصر (٣-٥ كلمات)
        2. يلخص الهدف الرئيسي
        3. يحافظ على المعنى الأساسي
        4. يكون باللهجة السعودية

        رد بالعنوان فقط بدون أي إضافات.
        """
        
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Task summary error: {e}")
        return text

def add_task(text):
    try:
        # Get task summary first
        task_summary = extract_task_summary(text)
        dt = parse_date_arabic(text) or datetime.now()
        
        emb = sentence_model.encode([task_summary])[0]
        tasks.append({"text": task_summary, "time": dt})
        embeddings.append(emb)
        index.add(np.array([emb]))
        save_tasks()
        
        print(f"✅ سجلت: {task_summary} @ {dt}")
        speak(f"تم تسجيل المهمة: {task_summary}")
    except Exception as e:
        print(f"Add task error: {e}")
        speak("حصل خطأ في تسجيل المهمة")

def delete_task(user_input):
    try:
        if not tasks:
            speak("ما عندك مهام عشان أحذفها.")
            return

        q_emb = sentence_model.encode([user_input])[0]
        _, I = index.search(np.array([q_emb]), 1)
        idx = I[0][0]

        deleted = tasks.pop(idx)
        embeddings.pop(idx)
        
        index.reset()
        if embeddings:
            index.add(np.array(embeddings))

        save_tasks()
        speak(f"تم حذف المهمة: {deleted['text']}")
    except Exception as e:
        print(f"Delete task error: {e}")
        speak("حصل خطأ في حذف المهمة")

def search_tasks(query, k=5):
    try:
        if not tasks:
            return []
        dt = parse_date_arabic(query)
        if dt:
            return [t for t in tasks if abs((t["time"].date() - dt.date()).days) <= 1]
        q_emb = sentence_model.encode([query])[0]
        _, I = index.search(np.array([q_emb]), k)
        return [tasks[i] for i in I[0] if i < len(tasks)]
    except Exception as e:
        print(f"Search tasks error: {e}")
        return []

def chat_response(user_input, related_tasks):
    try:
        if related_tasks:
            context = "\n".join([f"- {t['text']} (موعد: {t['time'].strftime('%Y-%m-%d %H:%M')})" for t in related_tasks])
        else:
            context = "ما عندك مهام مسجلة يا حلو."

        prompt = f"""أنت مساعد شخصي باللهجة السعودية.
        تحدث بأسلوب ودي ولطيف.
        
        إذا فهمت من سياق الكلام أن المستخدم يريد إنهاء المحادثة:
        1. رد بعبارة وداع مناسبة
        2. أضف في نهاية ردك السطر التالي: <close_conversation>

        السياق: {context}
        المستخدم: {user_input}

        رد:"""

        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Chat response error: {e}")
        return "حصل خطأ في الرد"

