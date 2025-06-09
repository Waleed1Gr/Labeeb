
WAKE_WORD = "لبيب"

SYSTEM_PROMPT = """
أنت مساعد ذكي يتحدث باللهجة السعودية. وظيفتك هي استقبال أوامر المستخدم ومعالجتها بدقة وبطريقة ودودة.
عندما يسمع كلمة الاستيقاظ "لبيب"، استجب للمستخدم وابدأ تنفيذ الأمر.
"""

def is_wake_word(text):
    return WAKE_WORD in text

def build_llm_prompt(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    return messages
