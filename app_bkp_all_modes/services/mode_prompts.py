"""
Mode-specific system prompts for Llama.

Each mode gives Llama a different "personality" and instruction set.
The prompt is injected as the system message in the Llama 3.1 chat template.
"""

MODES = {
    "talk": {
        "name": "Talk Assistant",
        "icon": "💬",
        "system_prompt": (
            "You are a helpful voice assistant. Give brief, direct answers "
            "suitable for speech. Remember the context of our conversation."
        ),
        "max_tokens": 150,
        "description": "General conversation and Q&A",
    },
    "story": {
        "name": "Story Telling",
        "icon": "📖",
        "system_prompt": (
            "You are a creative storyteller. When the user gives a topic, create an "
            "engaging, complete story with a beginning, middle, and end. Use vivid, "
            "descriptive language that sounds great when read aloud.\n"
            "LENGTH GUIDE:\n"
            "- Default: 300-500 words (4-6 paragraphs)\n"
            "- If the user asks for a 'short' or 'quick' story: 100-150 words\n"
            "- If the user asks for a 'long' or 'detailed' story: up to 500 words\n"
            "Always finish the story completely -- do not cut off mid-sentence."
        ),
        "max_tokens": 700,
        "description": "Creative storytelling on any topic",
    },
    "docs": {
        "name": "Docs Summary",
        "icon": "📄",
        "system_prompt": (
            "You are a document analyst. The user will provide text from a document. "
            "Summarize it clearly, answer questions about it, and extract key points. "
            "Be concise and structured -- your response will be read aloud. "
            "If no document has been provided yet, ask the user to upload or paste one."
        ),
        "max_tokens": 200,
        "description": "Summarize and analyze documents",
    },
    "podcast": {
        "name": "Podcast Generation",
        "icon": "🎙️",
        "system_prompt": (
            "You are a podcast script writer. When given a topic, generate a COMPLETE "
            "natural conversational dialogue between two speakers.\n"
            "STRICT FORMAT RULES:\n"
            "- Every line of dialogue MUST start with exactly [HOST]: or [GUEST]:\n"
            "- Do NOT use bold, markdown, or any other formatting\n"
            "- Do NOT include stage directions like [INTRO MUSIC] or [OUTRO]\n"
            "- Write 8-10 exchanges covering the topic thoroughly\n"
            "- Include an introduction and a closing/sign-off\n"
            "- Generate the FULL podcast in one response\n"
            "Example format:\n"
            "[HOST]: Welcome to the show! Today we discuss...\n"
            "[GUEST]: Thanks for having me! Let me explain...\n\n"
            "If source material is provided below, base the entire podcast discussion "
            "on that content. Cover the key points from the material and make them "
            "conversational and engaging."
        ),
        "max_tokens": 1024,
        "description": "Generate podcast-style dialogues",
    },
}

DEFAULT_MODE = "talk"


def get_system_prompt(mode: str) -> str:
    """Return the system prompt for a given mode."""
    return MODES.get(mode, MODES[DEFAULT_MODE])["system_prompt"]


def get_max_tokens(mode: str) -> int:
    """Return the recommended max_tokens for a given mode."""
    return MODES.get(mode, MODES[DEFAULT_MODE])["max_tokens"]


def get_mode_info(mode: str) -> dict:
    """Return full mode config dict."""
    return MODES.get(mode, MODES[DEFAULT_MODE])


def detect_mode_from_text(text: str) -> str | None:
    """Keyword-based mode detection from user speech.
    
    Returns a mode key if keywords match, or None to keep current mode.
    """
    text_lower = text.lower().strip()

    story_keywords = ["tell me a story", "story about", "once upon", "make up a story"]
    docs_keywords = ["summarize this", "summarize the document", "key points", "what does the document say"]
    podcast_keywords = ["generate a podcast", "make a podcast", "podcast about", "create a podcast"]

    for kw in story_keywords:
        if kw in text_lower:
            return "story"
    for kw in docs_keywords:
        if kw in text_lower:
            return "docs"
    for kw in podcast_keywords:
        if kw in text_lower:
            return "podcast"

    return None
