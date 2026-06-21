"""
Prompt templates for the LLM reasoning module.

Contains system prompts for grounded and ungrounded question answering,
user-message templates, evidence formatting templates, and claim extraction
prompts.  All templates are plain strings with ``{placeholder}`` markers
for use with ``str.format()`` or f-string interpolation at call sites.
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_GROUNDED: str = (
    "You are a precise temporal question-answering system. You must answer "
    "the user's question using ONLY the temporal evidence provided below. "
    "Do not use any prior knowledge or make assumptions beyond what the "
    "evidence explicitly states.\n\n"
    "Rules:\n"
    "1. Base every factual claim in your answer on one or more pieces of "
    "   the provided evidence.\n"
    "2. After each factual claim, cite the evidence index numbers that "
    "   support it (e.g. [1], [3]).\n"
    "3. If the evidence is insufficient to answer the question, state: "
    "   'Insufficient evidence to answer this question.'\n"
    "4. Do not speculate, hedge, or invent information.\n"
    "5. When dates or timestamps are relevant, state them explicitly.\n"
    "6. Keep your answer concise and factual."
)

SYSTEM_PROMPT_UNGROUNDED: str = (
    "You are a knowledgeable question-answering system. Answer the "
    "user's question directly and concisely based on your training "
    "knowledge. If you are not confident in the answer, say so."
)

# ---------------------------------------------------------------------------
# User message template
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE: str = (
    "Question: {question}\n\n"
    "Temporal Evidence:\n"
    "{evidence}\n\n"
    "Provide a concise, evidence-based answer."
)

# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------

EVIDENCE_FORMAT_TEMPLATE: str = (
    "{index}. ({subject}, {predicate}, {object}, {timestamp})"
)

# ---------------------------------------------------------------------------
# Claim extraction prompt
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT: str = (
    "You are a precise information extraction system. Your task is to "
    "extract all temporal claims from the following text.\n\n"
    "A temporal claim is a factual assertion that involves a specific time, "
    "date, or temporal relationship. For each claim, extract:\n"
    "  - subject: the main entity performing or involved in the action\n"
    "  - predicate: the action or relationship\n"
    "  - object: the target entity or outcome\n"
    "  - timestamp: the date or time reference (in YYYY-MM-DD format if "
    "possible, otherwise as stated)\n\n"
    "Output ONLY a JSON array of objects. Each object must have exactly "
    'these keys: "subject", "predicate", "object", "timestamp".\n\n'
    "If no temporal claims are found, output an empty JSON array: []\n\n"
    "Do not include any text outside the JSON array. Do not wrap the "
    "output in markdown code fences."
)
