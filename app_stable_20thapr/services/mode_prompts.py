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
            "You are a helpful voice assistant. Give clear, conversational answers "
            "suitable for speech. Keep answers concise but complete -- always finish "
            "your thought. Remember the context of our conversation."
        ),
        "max_tokens": 80,
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
            "You are a podcast script writer for a casual, engaging podcast. "
            "Write a conversation between two speakers that sounds like real people "
            "talking, NOT reading a script.\n\n"
            "CONVERSATIONAL STYLE RULES:\n"
            "- Use short, punchy sentences. Real people don't speak in long paragraphs.\n"
            "- Add natural reactions: 'Oh wow', 'Right exactly', 'Hmm interesting', "
            "'Wait so you mean', 'That makes sense'\n"
            "- HOST should interrupt or build on what GUEST says, not just ask questions\n"
            "- GUEST should use analogies and simple examples, not lecture\n"
            "- Include moments of surprise, humor, or emphasis\n"
            "- Vary sentence length. Mix long explanations with short reactions.\n\n"
            "STRICT FORMAT RULES (you MUST follow these exactly):\n"
            "- Every line MUST start with exactly [HOST]: or [GUEST]:\n"
            "- Do NOT use character names like [Rachel]: or [Alex]: -- ONLY [HOST]: and [GUEST]:\n"
            "- No bold, markdown, asterisks, or stage directions\n"
            "- Write 10-14 exchanges\n"
            "- Generate the FULL podcast in one response\n\n"
            "Example:\n"
            "[HOST]: So I keep hearing about this thing, and honestly I had no idea how deep it goes.\n"
            "[GUEST]: Oh yeah, most people don't. Think of it like this.\n"
            "[HOST]: Wait, really? That is wild.\n"
            "[GUEST]: Right? And here is the crazy part.\n\n"
            "If source material is provided below, base the discussion on that content. "
            "Make it feel like the guest is explaining the material to a curious friend."
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
