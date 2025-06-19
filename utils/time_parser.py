# utils/time_parser.py - Updated with default times and time detection

import re
from datetime import datetime, timedelta
from typing import Tuple, Optional

def parse_date_arabic(text: str) -> Tuple[Optional[datetime], bool]:
    """
    Parses Arabic date expressions and returns a tuple of:
    - datetime object (or None if no date found)
    - boolean indicating if time was explicitly specified
    
    Examples:
      - "اليوم" → (today 00:00, False)
      - "بكرة" → (tomorrow 00:00, False) 
      - "بكرة الساعة 3" → (tomorrow 3:00, True)
      - "الجمعة الساعة 2:30" → (next Friday 2:30, True)
    """
    text = text.strip().lower()
    now = datetime.now()
    time_specified = False
    target_date = None
    target_time = (0, 0)  # Default: 00:00 (no time)
    
    # Check for explicit time first (various patterns)
    time_patterns = [
        r"الساعة (\d{1,2})([:٫،](\d{1,2}))?\s*(مساءً|مساء|صباحاً|صباح)?",
        r"(\d{1,2})([:٫،](\d{1,2}))?\s*(مساءً|مساء|صباحاً|صباح)",
    ]
    
    for pattern in time_patterns:
        time_match = re.search(pattern, text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(3) or 0)
            period = time_match.group(4) or time_match.group(5) if len(time_match.groups()) > 4 else None
            
            # Handle مساءً/مساء (evening/PM) - add 12 hours if not already in 24h format
            if period and ("مساء" in period) and hour < 12:
                hour += 12
            
            target_time = (hour, minute)
            time_specified = True
            break
    
    # Parse date expressions
    if "اليوم" in text:
        target_date = now.date()
    elif "بكرة" in text:
        target_date = (now + timedelta(days=1)).date()
    elif "بعد بكرة" in text:
        target_date = (now + timedelta(days=2)).date()
    elif "نهاية الأسبوع" in text:
        days_ahead = (4 - now.weekday()) % 7
        target_date = (now + timedelta(days=days_ahead)).date()
    elif match := re.search(r"(ال)?(جمعة|سبت|أحد|اثنين|ثلاثاء|اربعاء|خميس)( الجاية)?", text):
        weekdays = {
            "سبت": 5, "أحد": 6, "اثنين": 0,
            "ثلاثاء": 1, "اربعاء": 2, "خميس": 3, "جمعة": 4
        }
        delta = (weekdays[match.group(2)] - now.weekday()) % 7 or 7
        target_date = (now + timedelta(days=delta)).date()
    elif match := re.search(r"(\d{1,2}) (يناير|فبراير|مارس|ابريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|اكتوبر|نوفمبر|ديسمبر)", text):
        months = {
            "يناير": 1, "فبراير": 2, "مارس": 3, "ابريل": 4,
            "مايو": 5, "يونيو": 6, "يوليو": 7, "أغسطس": 8,
            "سبتمبر": 9, "اكتوبر": 10, "نوفمبر": 11, "ديسمبر": 12
        }
        day, month = int(match.group(1)), months[match.group(2)]
        year = now.year + ((datetime(now.year, month, day) < now) and 1)
        target_date = datetime(year, month, day).date()
    
    # If we found a date, combine with time
    if target_date:
        result_datetime = datetime.combine(target_date, datetime.min.time().replace(
            hour=target_time[0], 
            minute=target_time[1]
        ))
        return result_datetime, time_specified
    
    # Handle time-only expressions (for today)
    if time_match and not target_date:
        hour, minute = target_time
        result_datetime = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return result_datetime, True
    
    return None, False


# Backward compatibility - old function signature
def parse_date_arabic_legacy(text: str) -> datetime:
    """Legacy function for backward compatibility"""
    result, _ = parse_date_arabic(text)
    return result