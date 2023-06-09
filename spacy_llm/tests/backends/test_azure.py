# mypy: ignore-errors
import contextlib
import os
from typing import List

import pytest
import spacy

from spacy_llm.backends.rest.backend.azure import AZURE_ENV_PREFIX, AzureOpenAIBackend

from ..compat import has_azure_env


@contextlib.contextmanager
def stash_env_vars(env_vars: List[str]):
    """Helper context to recover temporarily changed env vars."""
    stashed_env_vars = {}
    for key in env_vars:
        val = os.getenv(key)
        if val is None:
            continue
        stashed_env_vars[key] = val
        del os.environ[key]
    try:
        yield
    finally:
        for key, val in stashed_env_vars.items():
            if val is not None:
                os.environ[key] = val


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
def test_azure_bad_credential_failure():
    """Test error handling for incorrect credentials."""
    with stash_env_vars(["AZURE_OPENAI_API_KEY"]):
        os.environ["AZURE_OPENAI_API_KEY"] = "abcd1234wxyz"
        with pytest.raises(ValueError, match="Access denied"):
            AzureOpenAIBackend(
                config={"model": "gpt-3.5-turbo", "temperature": 0.3},
                strict=False,
                max_tries=10,
                interval=5.0,
                max_request_time=20,
            )


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
def test_azure_missing_credentials_failure():
    """Test error handling for no credentials."""
    with stash_env_vars(["AZURE_OPENAI_API_KEY"]):
        with pytest.raises(ValueError, match="Could not find the API key"):
            AzureOpenAIBackend(
                config={"model": "gpt-3.5-turbo", "temperature": 0.3},
                strict=False,
                max_tries=10,
                interval=5.0,
                max_request_time=20,
            )


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
@pytest.mark.parametrize(
    "parameter",
    ["resource", "deployment", "api_version"],
)
def test_azure_missing_parameter_failure(parameter):
    """Test error handling for missing parameters."""
    env_var = AZURE_ENV_PREFIX + parameter.upper()
    with stash_env_vars([env_var]):
        with pytest.raises(ValueError, match=f"Could not find parameter {parameter}"):
            AzureOpenAIBackend(
                config={"model": "gpt-3.5-turbo", "temperature": 0.3},
                strict=False,
                max_tries=10,
                interval=5.0,
                max_request_time=20,
            )


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
def test_azure_api_bad_request():
    """Test error handling a bad api request (bad parameter)."""
    rest = AzureOpenAIBackend(
        config={"model": "gpt-3.5-turbo", "temperature": 1000.0},
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )
    with pytest.raises(ValueError, match="Request to Azure OpenAI API failed"):
        rest(prompts=["this is a prompt"])


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
def test_azure_api_valid_request():
    """Test a valid request actually runs."""
    rest = AzureOpenAIBackend(
        config={"model": "gpt-3.5-turbo", "temperature": 0.3},
        strict=False,
        max_tries=10,
        interval=5.0,
        max_request_time=20,
    )
    prompts = ["Is this an insult or a complement?  Sam Malone has bad hair."] * 3
    completions = rest(prompts=prompts)
    for completion in completions:
        assert isinstance(completion, str)


@pytest.mark.skipif(has_azure_env is False, reason="Azure OpenAI API env vars not set")
@pytest.mark.external
def test_azure_spacy_integration():
    """Test the backend is properly registered and actually runs in a spaCy pipeline."""
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        config={
            "task": {
                "@llm_tasks": "spacy.TextCat.v2",
                "labels": "COMPLEMENT,INSULT",
                "examples": None,
                "exclusive_classes": True,
                "normalizer": {"@misc": "spacy.LowercaseNormalizer.v1"},
            },
            "backend": {
                "@llm_backends": "spacy.REST.v1",
                "api": "Azure",
                "config": {"model": "gpt-3.5-turbo", "temperature": 0.3},
            },
        },
    )
    doc = nlp("Sam Malone has bad hair.")
    assert doc.cats["INSULT"] == 1.0 and doc.cats["COMPLEMENT"] == 0.0
