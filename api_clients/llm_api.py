from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in .env or environment)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def classify_input(user_input: str) -> str:
    """
    Classify user input into one of:
      - 'تسجيل'       (add a task)
      - 'تذكير'       (remind/search tasks)
      - 'حذف'         (delete a single task)
      - 'حذف_الكل'    (delete all tasks)
      - 'غير'         (chat/general)
    """
    try:
        prompt = f"""المستخدم قال باللهجة السعودية: "{user_input}"

هل هذا طلب:
- تسجيل مهمة جديدة؟ (أجب: تسجيل)
- تذكير/استفسار عن المهام؟ (أجب: تذكير)
- حذف مهمة واحدة؟ (أجب: حذف)
- حذف جميع المهام؟ (أجب: حذف_الكل)
- أو شيء ثاني؟ (أجب: غير)

رد بكلمة واحدة فقط: تسجيل أو تذكير أو حذف أو حذف_الكل أو غير."""
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Classify input error: {e}")
        return "غير"


def chat_response(user_input: str, related_tasks: list) -> str:
    """
    Generate a conversational response based on user input and related tasks.
    If a close-conversation intent appears, it should include <close_conversation>.
    """
    try:
        if related_tasks:
            context = "\n".join([
                f"- {t['text']} (موعد: {t['time'].strftime('%Y-%m-%d %H:%M')})"
                for t in related_tasks
            ])
        else:
            context = "ما عندك مهام مسجلة يا حلو."

        prompt = f"""أنت مساعد شخصي باللهجة السعودية، هدفك هو الرد باختصار وبشكل طبيعي ومرح على المستخدم.

لو شعرت أن المستخدم يقصد إنهاء المحادثة (زي أنه يقول: خلاص، شكراً، مع السلامة، ما عاد أبي أتكلم)، رد بشكل مهذب برسالة وداع، وارجع لي السطر:

<close_conversation>

وإذا ما كان ينهي، فقط رد طبيعي من دون هذا السطر.

السياق:
{context}

المستخدم قال: {user_input}

رد:
"""
        res = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Chat response error: {e}")
        return "حصل خطأ في الرد"
