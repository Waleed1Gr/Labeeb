# api_clients/email_api.py

import os
from dotenv import load_dotenv
import yagmail

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

if not EMAIL_USER or not EMAIL_PASS or not EMAIL_TO:
    raise ValueError("Please set EMAIL_USER, EMAIL_PASS, and EMAIL_TO in your .env file")

def send_email(subject: str, body: str) -> bool:
    """
    Send an email with the given subject and body to the predefined recipient.

    Returns True if email sent successfully, else False.
    """
    try:
        # إنشاء الاتصال داخل الدالة (لتفادي مشكلة قراءة ملف التهيئة)
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        yag.send(to=EMAIL_TO, subject=subject, contents=body)
        print("📧 Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False