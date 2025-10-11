from __future__ import annotations

from typing import Optional


def system_date_expert() -> str:
    return "You are a date extraction expert."


def system_data_extractor() -> str:
    return "You are an expert data extractor. Output JSON only."


def global_date_user_prompt(header_text: str) -> str:
    return (
        "From the following text, which is the header of a document, extract the single, primary date "
        "of the meeting. Return JSON with one key, 'meeting_date'. If no date is found, return null.\n\n"
        f"TEXT: {header_text}"
    )


def adaptive_chunk_user_prompt(chunk_content: str, chunk_id: int, global_date_hint: Optional[str]) -> str:
    date_prompt = (
        f"The primary date for the entire document is likely {global_date_hint}. "
        if global_date_hint
        else ""
    )
    date_prompt += (
        "For each utterance, determine its specific date. If a date is mentioned in the chunk, use it. "
        "If no date is mentioned, use the primary document date if available, otherwise use null."
    )
    base = (
        f"{date_prompt}\n\nCHUNK {chunk_id}: Extract meeting data from this text.\n\nCONTENT:\n"
        f"{chunk_content[:8000]}\n\n"
        "Look for: Timestamps, Speaker names, and what each person said (utterances)."
        "\n\nReturn JSON array ONLY of objects with keys \"date\", \"timestamp\", \"speaker\", \"utterance\"."
    )
    return base


def single_shot_system() -> str:
    return (
        "You are an expert data extraction assistant. Your task is to analyze the provided text "
        "from a meeting transcript document and extract every utterance. For each utterance, you must identify "
        "the speaker, the time, and the date of the utterance. Format your output as a single JSON object with a key 'transcript', "
        "which contains a list of objects. Each object in the list must have the following keys with this exact capitalization: 'Date', 'Timestamp', 'Speaker', 'Utterance'. "
        "If you cannot find a value for a key, use a null value."
    )


def single_shot_user(raw_text: str) -> str:
    return raw_text


def chunk_system() -> str:
    return (
        "You are an expert data extraction assistant. The user will provide a chunk of text from a larger meeting transcript. "
        "Your task is to find and extract every distinct utterance from this chunk. For each utterance, you must identify the speaker, the time, and the utterance text. "
        "Format your output as a single JSON object with a key 'utterances', which contains a list of objects. "
        "Each object must have the following keys with this exact capitalization: 'Timestamp', 'Speaker', 'Utterance'."
    )

