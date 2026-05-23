"""LLM provider abstraction supporting Ollama, OpenAI, and Anthropic.

Public surface re-exports the same names that lived in the old
``halal_trader.core.llm`` module so callers don't need to change.
"""

from halal_trader.core.llm.anthropic import AnthropicLLM
from halal_trader.core.llm.base import BaseLLM, _clean_json_body, strip_thinking
from halal_trader.core.llm.factory import (
    _create_single_llm,
    create_classifier_llm,
    create_llm,
)
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.ollama import OllamaLLM
from halal_trader.core.llm.openai import OpenAILLM

__all__ = [
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "FallbackLLM",
    "create_llm",
    "create_classifier_llm",
    "_create_single_llm",
    "strip_thinking",
    "_clean_json_body",
]
