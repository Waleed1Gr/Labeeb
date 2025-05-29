import json
import numpy as np
from datetime import datetime, timedelta
from utils.config import DATA_TASKS_FILE
from audio.speak import speak
from utils.config import sentence_model
import re
from utils.config import client


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
    elif match := re.search(
        r"(ال)?(جمعة|سبت|أحد|اثنين|ثلاثاء|اربعاء|خميس)( الجاية)?", text
    ):
        weekdays = {
            "سبت": 5,
            "أحد": 6,
            "اثنين": 0,
            "ثلاثاء": 1,
            "اربعاء": 2,
            "خميس": 3,
            "جمعة": 4,
        }
        delta = (weekdays[match.group(2)] - now.weekday()) % 7 or 7
        return now + timedelta(days=delta)
    elif match := re.search(
        r"(\d{1,2}) (يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر)",
        text,
    ):
        months = {
            "يناير": 1,
            "فبراير": 2,
            "مارس": 3,
            "ابريل": 4,
            "مايو": 5,
            "يونيو": 6,
            "يوليو": 7,
            "أغسطس": 8,
            "سبتمبر": 9,
            "اكتوبر": 10,
            "نوفمبر": 11,
            "ديسمبر": 12,
        }
        day, month = int(match.group(1)), months[match.group(2)]
        year = now.year + ((datetime(now.year, month, day) < now) and 1)
        return datetime(year, month, day)
    elif match := re.search(r"الساعة (\d{1,2})([:٫،](\d{1,2}))?", text):
        hour, minute = int(match.group(1)), int(match.group(3) or 0)
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return None


def save_tasks():
    with open(DATA_TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": t["text"], "time": t["time"].isoformat()} for t in tasks],
            f,
            ensure_ascii=False,
        )


def load_tasks():
    global tasks, embeddings, index
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


def add_task(text):
    dt = parse_date_arabic(text) or datetime.now()
    emb = sentence_model.encode([text])[0]
    tasks.append({"text": text, "time": dt})
    embeddings.append(emb)
    index.add(np.array([emb]))
    save_tasks()
    print(f"✅ سجلت: {text} @ {dt}")
    speak("تم تسجيل المهمة يا بطل!")


def delete_task(user_input):
    if not tasks:
        speak("ما عندك مهام عشان أحذفها.")
        return

    q_emb = sentence_model.encode([user_input])[0]
    _, I = index.search(np.array([q_emb]), 1)
    idx = I[0][0]

    deleted = tasks.pop(idx)
    embeddings.pop(idx)

    # أعادة بناء الفهرس بعد الحذف
    index.reset()
    if embeddings:
        index.add(np.array(embeddings))

    save_tasks()
    speak(f"تم حذف المهمة: {deleted['text']}")


def search_tasks(query, k=5):
    dt = parse_date_arabic(query)
    if dt:
        return [t for t in tasks if abs((t["time"].date() - dt.date()).days) <= 1]
    q_emb = sentence_model.encode([query])[0]
    _, I = index.search(np.array([q_emb]), k)
    return [tasks[i] for i in I[0]]


def classify_input(user_input):
    prompt = f"""المستخدم قال باللهجة السعودية: "{user_input}"

هل هذا طلب:
- تسجيل مهمة جديدة؟ (أجب: تسجيل)
- تذكير/استفسار عن المهام؟ (أجب: تذكير)
- حذف مهمة؟ (أجب: حذف)
- أو شيء ثاني؟ (أجب: غير)

رد بكلمة واحدة فقط: تسجيل أو تذكير أو حذف أو غير.
"""
    res = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()


def chat_response(user_input, related_tasks):
    if related_tasks:
        context = "\n".join(
            [
                f"- {t['text']} (موعد: {t['time'].strftime('%Y-%m-%d %H:%M')})"
                for t in related_tasks
            ]
        )
    else:
        context = "ما عندك مهام مسجلة يا حلو."
    prompt = f"""أنت مساعد شخصي ترد باللهجة السعودية.\nالمهام:\n{context}\nالمستخدم قال: {user_input}\nرد عليه باختصار وخفة دم و رد مع الشدة والحركات مع الاحرف."""
    res = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return res.choices[0].message.content.strip()
