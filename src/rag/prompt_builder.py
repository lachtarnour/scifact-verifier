"""
Prompt construction for claim verification.

Builds a system prompt and a user prompt containing the claim
and retrieved abstracts. Works with any LLM backend.
"""

from typing import Dict, List

from src.data.load_scifact import CorpusType


SYSTEM_PROMPT = """\
You are an expert scientific fact-checker specialising in biomedical literature.
Your task is to verify scientific claims against retrieved peer-reviewed evidence.

Given a claim and a set of retrieved abstracts, provide:
1. A verdict: SUPPORTED, REFUTED, or NOT ENOUGH INFO
2. A confidence level: high, medium, or low
3. A clear explanation (2-4 sentences) citing specific evidence
4. The key evidence sentences from the abstracts

Rules:
- SUPPORTED: the abstracts provide direct evidence FOR the claim.
- REFUTED: the abstracts provide direct evidence AGAINST the claim.
- NOT ENOUGH INFO: the evidence is absent, tangential, or insufficient.
- Be conservative — prefer NOT ENOUGH INFO over hallucinating.
- Never introduce information not present in the provided abstracts."""


def build_user_prompt(
    claim: str,
    doc_ids: List[str],
    corpus: CorpusType,
) -> str:
    """Build the user prompt with the claim and retrieved abstracts."""
    lines = [f"CLAIM TO VERIFY:\n{claim}\n"]
    lines.append("RETRIEVED SCIENTIFIC ABSTRACTS:")
    lines.append("=" * 60)

    for i, doc_id in enumerate(doc_ids, start=1):
        doc = corpus.get(doc_id, {})
        title = doc.get("title", "Unknown title")
        abstract = doc.get("text", "No abstract available.")
        lines.append(f"\n[Document {i}] {title}")
        lines.append(f"Abstract: {abstract}")

    lines.append("\n" + "=" * 60)
    lines.append(
        "Based solely on the abstracts above, verify the claim. "
        "Provide your verdict, confidence, explanation, and key evidence."
    )
    return "\n".join(lines)
