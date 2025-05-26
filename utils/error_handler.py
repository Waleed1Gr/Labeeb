import logging
from functools import wraps
import time

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.error(f"Error: {e}, retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator