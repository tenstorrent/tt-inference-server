"""
Document Service - Extract text from PDFs, text files, and URLs.
Used by the Docs Summary mode to provide document context to Llama.
"""

import logging
import tempfile
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DocumentService:
    """Extracts text from uploaded documents and URLs."""

    def __init__(self):
        self.documents: Dict[str, str] = {}

    def store(self, session_id: str, text: str):
        """Store extracted document text for a session."""
        self.documents[session_id] = text
        logger.info(f"Stored document for session {session_id[:20]}... ({len(text)} chars)")

    def get(self, session_id: str) -> str | None:
        """Get stored document text for a session."""
        return self.documents.get(session_id)

    def clear(self, session_id: str):
        """Clear stored document for a session."""
        if session_id in self.documents:
            del self.documents[session_id]

    async def extract_from_upload(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from an uploaded file (PDF or plain text)."""
        try:
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                text = self._extract_pdf(file_data)
            elif ext in (".txt", ".md", ".csv", ".json", ".py", ".js", ".html"):
                text = file_data.decode("utf-8", errors="replace")
            else:
                text = file_data.decode("utf-8", errors="replace")

            text = text.strip()
            if not text:
                return {"error": "No text could be extracted from the file"}

            return {
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "filename": filename,
            }
        except Exception as e:
            logger.error(f"Document extraction error: {e}")
            return {"error": str(e)}

    async def extract_from_url(self, url: str) -> Dict[str, Any]:
        """Fetch a URL and extract readable text."""
        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (compatible; TTVoiceAssistant/1.0)"
            })
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            if not text:
                return {"error": "No readable text found at that URL"}

            return {
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
                "url": url,
            }
        except Exception as e:
            logger.error(f"URL extraction error: {e}")
            return {"error": str(e)}

    async def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Accept raw pasted text."""
        text = text.strip()
        if not text:
            return {"error": "Empty text provided"}
        return {
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def _extract_pdf(self, file_data: bytes) -> str:
        """Extract text from PDF bytes using PyPDF2."""
        import io
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return "[PDF support not available -- install PyPDF2]"

        reader = PdfReader(io.BytesIO(file_data))
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages)

    def get_context_for_prompt(self, session_id: str, max_chars: int = 6000) -> str | None:
        """Get document text truncated to fit in the prompt context window.
        
        With 4096 token context, ~6000 chars (~1500 tokens) is a good budget
        for the document, leaving room for system prompt + history + generation.
        """
        text = self.get(session_id)
        if not text:
            return None
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[... document truncated ...]"
        return text
