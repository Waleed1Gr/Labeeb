import re
from datetime import datetime, timedelta

def parse_date_arabic(text: str) -> datetime:
    """
    Parses Arabic date expressions like:
      - \"اليوم\" (today)
      - \"بكرة\" / \"بعد بكرة\"
      - weekday names (e.g. \"الجمعة الجاية\")
      - specific dates like \"15 يونيو\" etc.
      - times like \"الساعة 3:30\"
    Returns a Python datetime or None if no date keyword matched.
    """
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
        weekdays = {
            "سبت": 5, "أحد": 6, "اثنين": 0,
            "ثلاثاء": 1, "اربعاء": 2, "خميس": 3, "جمعة": 4
        }
        delta = (weekdays[match.group(2)] - now.weekday()) % 7 or 7
        return now + timedelta(days=delta)
    elif match := re.search(r"(\d{1,2}) (يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر)", text):
        months = {
            "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4,
            "مايو": 5, "يونيو": 6, "يوليو": 7, "أغسطس": 8,
            "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12
        }
        day, month = int(match.group(1)), months[match.group(2)]
        year = now.year + ((datetime(now.year, month, day) < now) and 1)
        return datetime(year, month, day)
    elif match := re.search(r"الساعة (\d{1,2})([:٫،](\d{1,2}))?", text):
        hour, minute = int(match.group(1)), int(match.group(3) or 0)
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return None
