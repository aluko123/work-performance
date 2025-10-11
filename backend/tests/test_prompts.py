import re

from backend.prompts import (
    # extraction
    system_date_expert,
    system_data_extractor,
    global_date_user_prompt,
    adaptive_chunk_user_prompt,
    single_shot_system,
    single_shot_user,
    chunk_system,
    # rag
    answer_system,
    answer_user_template,
    metadata_system,
    metadata_user_template,
)


def test_system_date_expert():
    assert system_date_expert() == "You are a date extraction expert."


def test_system_data_extractor():
    out = system_data_extractor()
    assert "expert data extractor" in out
    assert "Output JSON only" in out


def test_global_date_user_prompt_includes_header_and_instruction():
    header = "Team Meeting — March 10, 2024 — Agenda"
    out = global_date_user_prompt(header)
    assert "Return JSON" in out
    assert "'meeting_date'" in out
    assert header in out


def test_adaptive_chunk_user_prompt_with_and_without_hint():
    content = "09:00 Alice: Let's start. 09:05 Bob: Updates."
    # With global date hint
    out_with = adaptive_chunk_user_prompt(content, chunk_id=3, global_date_hint="2024-03-10")
    assert "CHUNK 3" in out_with
    assert "likely 2024-03-10" in out_with
    assert "Return JSON array ONLY" in out_with
    for key in ["date", "timestamp", "speaker", "utterance"]:
        assert f'"{key}"' in out_with

    # Without global date hint
    out_without = adaptive_chunk_user_prompt(content, chunk_id=1, global_date_hint=None)
    assert "CHUNK 1" in out_without
    assert "likely" not in out_without  # no hint text
    assert "Return JSON array ONLY" in out_without


def test_single_shot_system_contains_required_keys():
    sys = single_shot_system()
    # Mentions transcript, JSON and required top-level key
    assert "single JSON object" in sys
    assert "'transcript'" in sys
    # Mentions required fields with capitalization
    for key in ["Date", "Timestamp", "Speaker", "Utterance"]:
        assert f"'{key}'" in sys


def test_single_shot_user_passthrough():
    raw = "Some raw transcript text"
    assert single_shot_user(raw) == raw


def test_chunk_system_contains_required_keys():
    sys = chunk_system()
    assert "single JSON object" in sys
    assert "'utterances'" in sys
    for key in ["Timestamp", "Speaker", "Utterance"]:
        assert f"'{key}'" in sys


def test_answer_system_modes():
    base = answer_system(json_mode=False)
    assert "evidence-backed performance insights" in base
    assert "JSON object" not in base  # only in json_mode

    json_mode = answer_system(json_mode=True)
    for k in ["bullets", "metrics_summary", "follow_ups", "source_ids"]:
        assert k in json_mode
    assert "single JSON object" in json_mode


def test_answer_user_template_placeholders():
    tmpl = answer_user_template()
    for ph in ["{question}", "{analysis_type}", "{aggregates}", "{citations}", "{valid_source_ids}"]:
        assert ph in tmpl


def test_metadata_system_and_template():
    ms = metadata_system()
    assert "Return ONLY a single JSON object" in ms
    for k in ["bullets", "metrics_summary", "follow_ups", "source_ids"]:
        assert k in ms

    mt = metadata_user_template()
    for ph in ["{question}", "{answer}", "{citations}", "{valid_source_ids}"]:
        assert ph in mt

