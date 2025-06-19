# utils/number_normalizer.py

import re

# Map single digits to Arabic words
_DIGIT_MAP = {
    '0': ' صفر ',
    '1': ' واحد ',
    '2': ' إثنين ',
    '3': ' ثلاثة ',
    '4': ' أربعة ',
    '5': ' خمسة ',
    '6': ' ستة ',
    '7': ' سبعة ',
    '8': ' ثمانية ',
    '9': ' تسعة ',
}

_FLOAT_RE = re.compile(r'(\d+)\.(\d+)')

def normalize_numbers(text: str) -> str:
    """
    Replace occurrences of floats like 0.0, 12.34 with
    'digit digit ... فاصلة digit digit ...' in Arabic.
    """
    def _float_to_words(m):
        int_part, frac_part = m.group(1), m.group(2)
        int_words  = ' '.join(_DIGIT_MAP[d] for d in int_part)
        frac_words = ' '.join(_DIGIT_MAP[d] for d in frac_part)
        return f"{int_words} فاصلة {frac_words}"

    return _FLOAT_RE.sub(_float_to_words, text)