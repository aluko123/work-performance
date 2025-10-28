"""
Sanitization utilities to remove interpretive language from LLM outputs.
"""
import re
from typing import Dict


# Forbidden word patterns and their replacements
FORBIDDEN_PATTERNS: Dict[str, str] = {
    r'\bindicating\b': 'with',
    r'\bindicates\b': 'is',
    r'\bsuggesting\b': 'stating',
    r'\bsuggests\b': 'states',
    r'\breflecting\b': 'with',
    r'\breflects\b': 'shows',
    r'\beffective\b': '',
    r'\bongoing\b': 'continued',
    r'\bproactive\b': '',
    r'\bpositive\b': '',
    # Remove empty emphasis phrases
    r',\s+indicating\s+\w+\s+\w+\s+\w+\.': '.',
    r',\s+suggesting\s+\w+\s+\w+\s+\w+\.': '.',
}


def sanitize_text(text: str) -> str:
    """
    Remove interpretive language from text while preserving meaning.
    
    Args:
        text: Raw text from LLM
        
    Returns:
        Sanitized text with interpretive words removed/replaced
    """
    if not text:
        return text
    
    result = text
    
    # Apply pattern replacements
    for pattern, replacement in FORBIDDEN_PATTERNS.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Clean up double spaces and orphaned commas
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r',\s*\.', '.', result)
    result = re.sub(r'\s+,', ',', result)
    
    return result.strip()


def sanitize_bullets(bullets: list) -> list:
    """Sanitize a list of bullet points."""
    return [sanitize_text(b) for b in bullets if b]


def sanitize_metrics_summary(summary: list) -> list:
    """Sanitize metrics summary entries."""
    sanitized = []
    for item in summary:
        if isinstance(item, dict):
            sanitized.append({
                k: sanitize_text(v) if isinstance(v, str) else v
                for k, v in item.items()
            })
        elif isinstance(item, str):
            sanitized.append(sanitize_text(item))
        else:
            sanitized.append(item)
    return sanitized
