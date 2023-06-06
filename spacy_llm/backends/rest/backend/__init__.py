from typing import Dict, Type

from . import azure, base, noop, openai

supported_apis: Dict[str, Type[base.Backend]] = {
    "Azure": azure.AzureOpenAIBackend,
    "OpenAI": openai.OpenAIBackend,
    "NoOp": noop.NoOpBackend,
}

__all__ = ["azure", "base", "openai", "noop", "supported_apis"]
