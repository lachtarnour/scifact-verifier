"""
Local LLM via Ollama — default backend, no API key needed.
"""

import json
import time
from typing import Dict, Generator, List

import requests

from src.data.load_scifact import CorpusType
from src.rag.base_generator import BaseGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.utils import config, get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 5
_RETRY_DELAY = 3

_OLLAMA_OPTIONS = {
    "temperature": 0,
}


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

    def _messages(self, claim: str, doc_ids: List[str], corpus: CorpusType) -> list:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(claim, doc_ids, corpus)},
        ]

    def _post(self, body: dict, stream: bool = False) -> requests.Response:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._api_url("/api/chat"),
                    json=body,
                    stream=stream,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.HTTPError) as e:
                if attempt == _MAX_RETRIES:
                    raise
                logger.info("Ollama unavailable, retry %d/%d", attempt, _MAX_RETRIES)
                time.sleep(_RETRY_DELAY)

    def generate(
        self,
        claim: str,
        doc_ids: List[str],
        corpus: CorpusType,
    ) -> Dict:
        logger.info("Ollama %s — %s", self.model, claim[:60])

        resp = self._post({
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": _OLLAMA_OPTIONS,
            "messages": self._messages(claim, doc_ids, corpus),
        })

        content = resp.json().get("message", {}).get("content", "")
        logger.info("Response: %d chars", len(content))

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "verdict": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "explanation": "Model did not return valid JSON.",
                "cited_docs": [],
                "evidence": [],
                "raw_output": content,
            }

    def stream(
        self,
        claim: str,
        doc_ids: List[str],
        corpus: CorpusType,
    ) -> Generator[str, None, None]:
        logger.info("Streaming %s — %s", self.model, claim[:60])

        resp = self._post(
            {
                "model": self.model,
                "stream": True,
                "format": "json",
                "options": _OLLAMA_OPTIONS,
                "messages": self._messages(claim, doc_ids, corpus),
            },
            stream=True,
        )

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                token = json.loads(line).get("message", {}).get("content", "")
                if token:
                    yield token
            except json.JSONDecodeError:
                continue
