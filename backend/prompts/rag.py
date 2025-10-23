from __future__ import annotations

def answer_system(json_mode: bool = False) -> str:
    base = (
        "You are an assistant producing concise, evidence-backed performance insights. "
        "CRITICAL GROUNDING RULES:\n"
        "1. ONLY use information explicitly present in the provided citations and aggregates.\n"
        "2. Do NOT use external knowledge, assumptions, or general statements not supported by the data.\n"
        "3. If the question cannot be answered with the provided context, say 'The available data does not contain enough information to answer this question.'\n"
        "4. Every claim must be directly traceable to a specific citation or aggregate metric.\n"
        "5. Use exact quotes or paraphrases from citations when possible.\n"
        "6. If aggregates show a metric, reference it with specific numbers (e.g., 'average score of 0.85').\n"
        "7. Never generalize beyond what the data explicitly shows.\n\n"
        "First, provide a direct, narrative answer to the user's question based ONLY on the provided context. "
        "Your answer must be fully grounded in the data and citations provided."
    )
    if json_mode:
        base += (
            " After your narrative answer, you MUST provide a JSON object with keys: "
            "'answer', 'bullets', 'metrics_summary', 'follow_ups', and 'source_ids'. "
            "You MUST include 'source_ids' as a list of integers that reference the provided citations' source_id values that were actually used. "
            "Do NOT invent IDs. Only include source_ids for citations that directly support your answer. "
            "If no citations support the answer, use an empty list. "
            "The final output MUST be a single JSON object containing all required keys."
        )
    return base


def answer_user_template() -> str:
    return (
        "Question: {question}\n"
        "Analysis type: {analysis_type}\n\n"
        "AVAILABLE DATA:\n"
        "Aggregates (sample size, averages): {aggregates}\n"
        "Citations (verbatim from database): {citations}\n"
        "Valid citation source_ids: {valid_source_ids}\n\n"
        "INSTRUCTIONS:\n"
        "- Answer using ONLY the information in the aggregates and citations above.\n"
        "- Reference specific data points (speakers, dates, scores, quotes).\n"
        "- If data is insufficient, explicitly state what is missing.\n"
        "Constraints: <=120 words in 'answer'; 3-5 bullets; 2-4 follow_ups."
    )


def metadata_system() -> str:
    return (
        "Given the user's question and the provided answer, generate the associated metadata. "
        "CRITICAL: Only include source_ids for citations that were ACTUALLY used to produce the answer. "
        "Do NOT include citations that were merely provided but not referenced. "
        "Ensure all bullets and metrics are grounded in the provided citations. "
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

