"""
Generator factory.

If ANTHROPIC_API_KEY is set → ClaudeGenerator
Otherwise → OllamaGenerator (local, no key needed)
"""

from src.rag.base_generator import BaseGenerator
from src.config import config
from src.utils import get_logger

logger = get_logger(__name__)


def create_generator() -> BaseGenerator:
    """Create the appropriate generator based on config."""
    if config.ANTHROPIC_API_KEY:
        from src.rag.claude_generator import ClaudeGenerator
        logger.info("Generator: Claude (%s)", config.CLAUDE_MODEL)
        return ClaudeGenerator()

    from src.rag.ollama_generator import OllamaGenerator
    logger.info("Generator: Ollama (%s)", config.OLLAMA_MODEL)
    return OllamaGenerator()
