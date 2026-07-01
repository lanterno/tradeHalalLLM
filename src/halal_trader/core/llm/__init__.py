"""LLM abstraction — GLM-5.2 only (OpenAI-compatible endpoints).

Public surface re-exports the same names that lived in the old
``halal_trader.core.llm`` module so callers don't need to change.
"""

from halal_trader.core.llm.base import BaseLLM, _clean_json_body, strip_thinking
from halal_trader.core.llm.factory import create_classifier_llm, create_llm
from halal_trader.core.llm.fallback import FallbackLLM
from halal_trader.core.llm.glm import GLMLLM

__all__ = [
    "BaseLLM",
    "GLMLLM",
    "FallbackLLM",
    "create_llm",
    "create_classifier_llm",
    "strip_thinking",
    "_clean_json_body",
]
