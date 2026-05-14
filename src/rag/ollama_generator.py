"""
Local LLM generator via Ollama.

Default backend — no API key needed.
Requires Ollama running locally or in Docker (http://localhost:11434).
"""

import json
from typing import Dict, Generator, List

import requests

from src.data.load_scifact import CorpusType
from src.rag.base_generator import BaseGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.utils import config, get_logger

logger = get_logger(__name__)


class OllamaGenerator(BaseGenerator):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or config.OLLAMA_MODEL
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")

    def _api_url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    # ── Full generation ──────────────────────────────────────────

    def generate(
        self,
        claim: str,
        doc_ids: List[str],
        corpus: CorpusType,
    ) -> Dict:
        user_prompt = build_user_prompt(claim, doc_ids, corpus)

        logger.info("Ollama %s — %s", self.model, claim[:60])

        response = requests.post(
            self._api_url("/api/chat"),
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")

        logger.info("Response: %d chars", len(content))

        return {"content": content}

    # ── Streaming ────────────────────────────────────────────────

    def stream(
        self,
        claim: str,
        doc_ids: List[str],
        corpus: CorpusType,
    ) -> Generator[str, None, None]:
        user_prompt = build_user_prompt(claim, doc_ids, corpus)

        logger.info("Streaming %s — %s", self.model, claim[:60])

        response = requests.post(
            self._api_url("/api/chat"),
            json={
                "model": self.model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
            except json.JSONDecodeError:
                continue
