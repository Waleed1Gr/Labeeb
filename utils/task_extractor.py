# utils/task_extractor.py - Updated to handle time extraction better
from api_clients.llm_api import client
import re
from typing import Optional

def extract_task_content(text: str) -> str:
    """
    Extracts the meaningful task content from Arabic command sentences.
    Removes command prefixes while preserving the core task and timing information.
    
    Examples:
        "سجل مهمة أداء الاختبار بكرة" → "أداء الاختبار بكرة"
        "ذكرني اجتماع مع العميل الخميس الساعة 5 مساءً" → "اجتماع مع العميل الخميس الساعة 5 مساءً"
        "أضف مهمة شراء الكتب اليوم" → "شراء الكتب اليوم"
    """
    
    # Clean and normalize the text
    text = text.strip()
    
    # Define command prefixes to remove (order matters - longer patterns first)
    command_patterns = [
        r'^سجل\s+مهمة\s+',           # "سجل مهمة "
        r'^اضف\s+مهمة\s+',          # "اضف مهمة "
        r'^أضف\s+مهمة\s+',          # "أضف مهمة "
        r'^سجل\s+لي\s+مهمة\s+',     # "سجل لي مهمة "
        r'^اضف\s+لي\s+مهمة\s+',    # "اضف لي مهمة "
        r'^أضف\s+لي\s+مهمة\s+',    # "أضف لي مهمة "
        r'^ذكرني\s+',               # "ذكرني "
        r'^ذكرني\s+بـ\s*',          # "ذكرني بـ" or "ذكرني ب"
        r'^سجل\s+',                 # "سجل "
        r'^اضف\s+',                 # "اضف "
        r'^أضف\s+',                 # "أضف "
        r'^مهمة\s+',                # "مهمة " (if it starts with just this)
    ]
    
    # Apply each pattern to remove command prefixes
    extracted_text = text
    for pattern in command_patterns:
        extracted_text = re.sub(pattern, '', extracted_text, flags=re.IGNORECASE).strip()
        if extracted_text != text:  # If a pattern matched, break
            break
    
    # If nothing was extracted or text is too short, return original
    if not extracted_text or len(extracted_text.strip()) < 3:
        return text
    
    # Clean up any remaining artifacts
    extracted_text = re.sub(r'^\s*[،,]\s*', '', extracted_text)  # Remove leading commas
    extracted_text = re.sub(r'^\s*أن\s+', '', extracted_text)     # Remove "أن" if it starts
    extracted_text = re.sub(r'^\s*إن\s+', '', extracted_text)     # Remove "إن" if it starts
    
    return extracted_text.strip()



def extract_task_content_no_time(text: str) -> str:
    """
    Extracts task content and removes time information if no explicit time was mentioned.
    This is for display purposes when time_specified = False.
    
    Examples:
        "أداء الاختبار بكرة" → "أداء الاختبار بكرة" (keep date)
        "اجتماع مع العميل الخميس الساعة 5 مساءً" → "اجتماع مع العميل الخميس الساعة 5 مساءً" (keep time)
    """
    
    # Extract base content first
    base_content = extract_task_content(text)
    
    # Check if this contains explicit time markers
    time_patterns = [
        r"الساعة\s+\d{1,2}([:٫،]\d{1,2})?\s*(مساءً|مساء|صباحاً|صباح)?",
        r"\d{1,2}([:٫،]\d{1,2})?\s*(مساءً|مساء|صباحاً|صباح)",
    ]
    
    has_explicit_time = any(re.search(pattern, base_content) for pattern in time_patterns)
    
    if has_explicit_time:
        # Keep the time information since it was explicitly mentioned
        return base_content
    else:
        # No explicit time - just return the clean task content
        return base_content


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

def extract_task_with_context(text: str, conversation_history: list) -> str:
    """
    Extracts task content considering conversation history when references like
    "this" or "what we just talked about" are detected.
    
    Args:
        text: The user's command text
        conversation_history: List of recent conversation exchanges
        
    Returns:
        str: The extracted task content
    """
    # First try normal extraction
    extracted = extract_task_content(text)
    
    # Check for reference indicators in Arabic
    reference_words = [
        "هذا", "هذه", "ذلك", "تلك", "هذي", "هذا الشيء", "هذا الموضوع", 
        "اللي تكلمنا عنه", "اللي قلته", "اللي قلناه", 
        "موضوعنا", "كلامنا", "المحادثة"
    ]
    
    has_reference = any(word in text.lower() for word in reference_words)
    
    if has_reference and conversation_history:
        # If the extracted content is too short or contains reference words
        if len(extracted.split()) <= 3 or any(word in extracted.lower() for word in reference_words):
            # Get the most recent exchange
            last_exchange = conversation_history[-1]
            
            # Create a prompt for the LLM that includes context from the conversation
            context_prompt = f"""
            المحادثة الأخيرة كانت:
            
            المستخدم: {last_exchange.get('user', '')}
            المساعد: {last_exchange.get('assistant', '')}
            
            ثم المستخدم قال: "{text}"
            
            استخرج عنوان مختصر للمهمة بناءً على موضوع المحادثة الأخيرة.
            """
            
            # Use the existing extract_task_summary function
            return extract_task_summary(context_prompt)
    
    return extracted
