# utils/text_cleaner.py
import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Strip whitespace
      - Normalize “fancy” quotes to straight quotes
      - Collapse multiple spaces into one
    """
    text = text.strip()
    text = text.replace("“", "\"").replace("”", "\"") \
               .replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text
