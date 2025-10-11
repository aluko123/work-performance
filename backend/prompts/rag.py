from __future__ import annotations

def answer_system(json_mode: bool = False) -> str:
    base = (
        "You are an assistant producing concise, evidence-backed performance insights. "
        "First, provide a direct, narrative answer to the user's question based on the provided context. "
        "Your answer should be grounded in the data and citations provided."
    )
    if json_mode:
        base += (
            " After your narrative answer, you MUST provide a JSON object with keys: "
            "'bullets', 'metrics_summary', 'follow_ups', and 'source_ids'. "
            "You MUST include 'source_ids' as a list of integers that reference the provided citations' source_id values. "
            "Do NOT invent IDs. If uncertain, pick the closest citation and include its source_id. "
            "The final output MUST be a single JSON object containing both 'answer' and the other keys."
        )
    return base


def answer_user_template() -> str:
    return (
        "Question: {question}\n"
        "Analysis type: {analysis_type}\n"
        "Aggregates (sample size, averages): {aggregates}\n"
        "Citations (for reference): {citations}\n"
        "Valid citation source_ids: {valid_source_ids}\n"
        "Constraints: <=120 words in 'answer'; 3-5 bullets; 2-4 follow_ups."
    )


def metadata_system() -> str:
    return (
        "Given the user's question and the provided answer, generate the associated metadata. "
        "Return ONLY a single JSON object with keys: 'bullets', 'metrics_summary', 'follow_ups', and 'source_ids'."
    )


def metadata_user_template() -> str:
    return (
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Citations (for reference): {citations}\n"
        "Valid citation source_ids: {valid_source_ids}\n"
        "Constraints: 3-5 bullets; 2-4 follow_ups; 'source_ids' must be a list of integers from the valid IDs."
    )

