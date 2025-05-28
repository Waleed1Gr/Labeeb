import os
import time
import faiss
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re

# ملفات التخزين
DATA_TASKS_FILE = "tasks.json"
DATA_INDEX_FILE = "tasks.index"

# إعدادات OpenAI
# client = OpenAI(api_key="")  # ضع مفتاح API الخاص بك هنا
# تحميل نموذج التعابير
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
dimension = 384

# المهام والـ FAISS
tasks = []
embeddings = []
index = faiss.IndexFlatL2(dimension)

# ✅ دالة تحليل التاريخ بالعربية
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
        days_until_friday = (4 - now.weekday()) % 7
        return now + timedelta(days=days_until_friday)
    elif match := re.search(r"(ال)?(جمعة|سبت|أحد|اثنين|ثلاثاء|اربعاء|خميس)( الجاية)?", text):
        weekdays = {
            "سبت": 5, "أحد": 6, "اثنين": 0, "ثلاثاء": 1,
            "اربعاء": 2, "خميس": 3, "جمعة": 4
        }
        target = weekdays[match.group(2)]
        delta = (target - now.weekday()) % 7
        delta = 7 if delta == 0 else delta
        return now + timedelta(days=delta)
    elif match := re.search(r"(\d{1,2}) (يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر)", text):
        day = int(match.group(1))
        month_map = {
            "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4, "مايو": 5,
            "يونيو": 6, "يوليو": 7, "أغسطس": 8, "سبتمبر": 9, "اكتوبر": 10,
            "نوفمبر": 11, "ديسمبر": 12
        }
        month = month_map[match.group(2)]
        year = now.year
        dt = datetime(year, month, day)
        if dt < now:
            dt = datetime(year + 1, month, day)
        return dt
    elif match := re.search(r"الساعة (\d{1,2})([:٫،](\d{1,2}))?", text):
        hour = int(match.group(1))
        minute = int(match.group(3)) if match.group(3) else 0
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None

# تخزين واسترجاع المهام
def save_tasks():
    tasks_serializable = [
        {"text": t["text"], "time": t["time"].isoformat()} for t in tasks
    ]
    with open(DATA_TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks_serializable, f, ensure_ascii=False, indent=2)

def load_tasks():
    global tasks, embeddings, index
    if os.path.exists(DATA_TASKS_FILE):
        with open(DATA_TASKS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        tasks = []
        embeddings.clear()
        index.reset()
        for t in loaded:
            dt = datetime.fromisoformat(t["time"])
            tasks.append({"text": t["text"], "time": dt})
            emb = model.encode([t["text"]])[0]
            embeddings.append(emb)
        if embeddings:
            index.add(np.array(embeddings))

def save_index():
    faiss.write_index(index, DATA_INDEX_FILE)

def load_index():
    global index
    if os.path.exists(DATA_INDEX_FILE):
        index = faiss.read_index(DATA_INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(dimension)

# ✅ تسجيل المهمة
def add_task(text):
    dt = parse_date_arabic(text)
    if dt is None:
        dt = datetime.now()

    emb = model.encode([text])[0]
    tasks.append({"text": text, "time": dt})
    embeddings.append(emb)
    index.add(np.array([emb]))
    save_tasks()
    save_index()
    print(f"✅ تم تسجيل المهمة مع التاريخ: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    نص المهمة: {text}")

# البحث عن المهام
def search_tasks(query, k=5):
    dt = parse_date_arabic(query)
    if dt:
        filtered = [t for t in tasks if abs((t["time"].date() - dt.date()).days) <= 1]
        if filtered:
            return filtered

    if not tasks:
        return []

    q_emb = model.encode([query])[0]
    _, I = index.search(np.array([q_emb]), k)
    return [tasks[i] for i in I[0]]

# تصنيف نوع الإدخال
def classify_input(user_input):
    prompt = f"""المستخدم قال باللهجة السعودية: "{user_input}"

هل هذا طلب:
- تسجيل مهمة جديدة؟ (أجب: تسجيل)
- تذكير/استفسار عن المهام؟ (أجب: تذكير)
- أو شيء ثاني؟ (أجب: غير)

رد بكلمة واحدة فقط: تسجيل أو تذكير أو غير.
"""
    res = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content.strip()

# محادثة المساعد
def chat(user_input, related_tasks):
    if related_tasks:
        context = "\n".join([f"- {t['text']} (موعد: {t['time'].strftime('%Y-%m-%d %H:%M')})" for t in related_tasks])
    else:
        context = "والله ما عندك شيء مسجل."

    prompt = f"""أنت مساعد شخصي باللهجة السعودية.

- كلامك مختصر بس فيه طرافة وخفة دم.
- ترد على المستخدم باللهجة السعودية.   
- لا تكثر كلام، بس تكفيك جملتين أو ثلاث.
المهام هذي:

{context}

المستخدم قال: {user_input}
رد عليه باللهجة السعودية و امزح شوي، لا تطول بالكلام."""
    
    res = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return res.choices[0].message.content.strip()
# المراقبة من ملف STT
def watch_file(file_path):
    print("🎤 جاهز لاستقبال أوامر STT من الملف...")
    last_text = ""
    load_tasks()
    load_index()

    while True:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text and text != last_text:
                print(f"\n📥 مدخل STT: {text}")
                last_text = text

                intent = classify_input(text)
                if intent == "تسجيل":
                    add_task(text)
                elif intent == "تذكير":
                    related = search_tasks(text)
                    reply = chat(text, related)
                    print("🤖:", reply)
                else:
                    print("❗ ما فهمت الأمر، حاول تعيد صياغة طلبك.")
        time.sleep(1)

if __name__ == "__main__":
    watch_file("stt_output.txt")
