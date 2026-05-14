"""
Prompt construction for scientific claim verification.
"""

from typing import List

from src.data.load_scifact import CorpusType


SYSTEM_PROMPT = """\
You are a scientific claim verifier. You assess biomedical claims against retrieved abstracts.

RULES:
- Use ONLY the provided abstracts. No outside knowledge.
- SUPPORTED: at least one abstract directly and explicitly supports the exact claim.
- REFUTED: at least one abstract directly and explicitly contradicts the exact claim.
- NOT ENOUGH INFO: the abstracts are indirectly related, discuss a different outcome or population, or do not explicitly address the claim.
- Same topic does NOT mean direct support. Related outcome does NOT mean exact claim.
- When in doubt, return NOT ENOUGH INFO.
- Do not infer beyond what the abstracts state.

CITATION RULES:
- "cited_docs" must contain document numbers from the prompt (e.g. [1, 3]). Do not use PubMed IDs.
- Only cite documents that directly address the verdict.
- Evidence sentences must be copied exactly from the abstracts. Do not paraphrase or invent.
- For SUPPORTED or REFUTED, you must cite at least one document and one evidence sentence.

Respond with valid JSON only, no extra text:
{
  "verdict": "SUPPORTED" | "REFUTED" | "NOT ENOUGH INFO",
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<2-3 sentences>",
  "cited_docs": [1, 3],
  "evidence": ["exact sentence from abstract"]
}"""


def build_user_prompt(
    claim: str,
    doc_ids: List[str],
    corpus: CorpusType,
) -> str:
    lines = [f"CLAIM: {claim}\n", "ABSTRACTS:"]

    for i, doc_id in enumerate(doc_ids, start=1):
        doc = corpus.get(doc_id, {})
        title = doc.get("title", "Unknown")
        abstract = doc.get("text", "")
        lines.append(f"\n[Document {i}] {title}")
        lines.append(abstract)

    lines.append("\nVerify the claim using ONLY the abstracts above. Respond with JSON only.")
    return "\n".join(lines)
