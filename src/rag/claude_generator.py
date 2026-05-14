"""
Claude API generator (optional).
Used only if ANTHROPIC_API_KEY is set.
"""

import json
from typing import Dict, Generator, List

import anthropic

from src.data.load_scifact import CorpusType
from src.rag.base_generator import BaseGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from src.utils import config, get_logger

logger = get_logger(__name__)


class ClaudeGenerator(BaseGenerator):
    def __init__(self, model: str | None = None):
        self.model = model or config.CLAUDE_MODEL
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def generate(
        self,
        claim: str,
        doc_ids: List[str],
        corpus: CorpusType,
    ) -> Dict:
        client = self._get_client()
        user_prompt = build_user_prompt(claim, doc_ids, corpus)

        logger.info("Claude %s — %s", self.model, claim[:60])

        response = client.messages.create(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        content = response.content[0].text
        logger.info("Response: %d tokens", response.usage.output_tokens)

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
        client = self._get_client()
        user_prompt = build_user_prompt(claim, doc_ids, corpus)

        logger.info("Streaming %s — %s", self.model, claim[:60])

        with client.messages.stream(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for token in stream.text_stream:
                yield token
