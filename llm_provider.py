"""
llm_provider.py
===============

Implements the Strategy Pattern for LLM inference so the rest of the
pipeline (generator.py, evaluator.py) never needs to know which underlying
model/API is actually being called.

Two concrete strategies are provided:

  - HuggingFaceProvider  : calls a model hosted on the Hugging Face
                           Inference API (e.g. a Hugging Face Space or a
                           serverless Inference Endpoint) via `requests`.
  - StandardAPIProvider  : calls any OpenAI-compatible chat-completions API
                           (OpenAI, Groq, or Google Gemini's OpenAI-compat
                           endpoint) using the official `openai` python
                           package.

Switching providers is done purely through environment variables / a config
dict -- no code changes required. See `get_llm_provider()` at the bottom.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider fails after retries are exhausted."""


class LLMProvider(ABC):
    """
    Abstract base class for any LLM inference backend.

    Concrete subclasses must implement `generate`, which takes a prompt
    (and optional system message) and returns the plain-text completion.
    """

    def __init__(self, model_name: str, max_retries: int = 3, retry_backoff: float = 2.0):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @abstractmethod
    def _call(self, prompt: str, system: Optional[str], temperature: float,
               max_tokens: int) -> str:
        """Subclasses implement the actual network call here."""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 800,
    ) -> str:
        """
        Generate text from the underlying LLM, with retry + backoff on
        transient failures (network errors, rate limits, 5xx responses).
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call(prompt, system, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001 - we want to catch & retry broadly
                last_error = exc
                wait = self.retry_backoff ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d) via %s: %s. Retrying in %.1fs...",
                    attempt, self.max_retries, self.__class__.__name__, exc, wait,
                )
                time.sleep(wait)

        raise LLMProviderError(
            f"{self.__class__.__name__} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


class HuggingFaceProvider(LLMProvider):
    """
    Calls a model hosted on the Hugging Face Inference API / Hugging Face
    Spaces (Inference Endpoints), using plain HTTP requests.

    Env vars used (if not passed explicitly):
        HF_API_TOKEN   -> Hugging Face access token
        HF_MODEL_ID    -> e.g. "mistralai/Mistral-7B-Instruct-v0.3"
        HF_API_URL     -> optional override; defaults to the standard
                          serverless inference URL for HF_MODEL_ID
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_token: Optional[str] = None,
        api_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        model_name = model_name or os.environ.get("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
        super().__init__(model_name=model_name, max_retries=max_retries)

        self.api_token = api_token or os.environ.get("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "HuggingFaceProvider requires an API token. Set HF_API_TOKEN "
                "as an environment variable or pass api_token explicitly."
            )

        self.api_url = api_url or os.environ.get(
            "HF_API_URL", f"https://api-inference.huggingface.co/models/{self.model_name}"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def _build_chat_text(self, prompt: str, system: Optional[str]) -> str:
        """Most HF text-generation-inference models expect a single text
        blob rather than a structured messages array, so we format a simple
        instruction-style prompt."""
        if system:
            return f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> str:
        payload = {
            "inputs": self._build_chat_text(prompt, system),
            "parameters": {
                "temperature": max(temperature, 0.01),
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        response = requests.post(
            self.api_url, headers=self.headers, data=json.dumps(payload), timeout=120
        )

        if response.status_code == 503:
            # Model is loading on HF infra -- worth a retry.
            raise LLMProviderError(f"Model still loading on Hugging Face: {response.text[:300]}")
        if response.status_code != 200:
            raise LLMProviderError(
                f"Hugging Face API error {response.status_code}: {response.text[:500]}"
            )

        data = response.json()

        # Response shape varies by model/task; handle the common cases.
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        if isinstance(data, dict) and "error" in data:
            raise LLMProviderError(f"Hugging Face API returned an error: {data['error']}")

        raise LLMProviderError(f"Unrecognized Hugging Face API response shape: {data}")


class StandardAPIProvider(LLMProvider):
    """
    Calls any OpenAI-compatible chat-completions endpoint using the official
    `openai` python package. Works out of the box with:
      - OpenAI            (base_url default, api key from OPENAI_API_KEY)
      - Groq              (base_url="https://api.groq.com/openai/v1")
      - Google Gemini     (base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    Env vars used (if not passed explicitly):
        STANDARD_API_KEY   -> API key for whichever provider you're using
        STANDARD_API_BASE  -> base_url override (leave unset for real OpenAI)
        STANDARD_MODEL     -> model name, e.g. "gpt-4o-mini", "llama-3.1-70b-versatile"
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        model_name = model_name or os.environ.get("STANDARD_MODEL", "gpt-4o-mini")
        super().__init__(model_name=model_name, max_retries=max_retries)

        api_key = api_key or os.environ.get("STANDARD_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "StandardAPIProvider requires an API key. Set STANDARD_API_KEY "
                "(or OPENAI_API_KEY) as an environment variable or pass api_key explicitly."
            )

        base_url = base_url or os.environ.get("STANDARD_API_BASE")  # None -> real OpenAI default

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for StandardAPIProvider. "
                "Install it with `pip install openai`."
            ) from e

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def _call(self, prompt: str, system: Optional[str], temperature: float,
              max_tokens: int) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(f"StandardAPIProvider call failed: {exc}") from exc

        if not completion.choices:
            raise LLMProviderError("StandardAPIProvider returned no choices in the response.")

        choice = completion.choices[0]
        message = choice.message
        content = (message.content or "").strip()

        # Some providers (notably free-tier / "thinking" models served
        # through OpenRouter, e.g. Tencent Hunyuan, DeepSeek-R1 style
        # models) put the model's output in a separate `reasoning` /
        # `reasoning_content` field and leave `content` empty, especially
        # when the reasoning trace consumes the whole token budget. Fall
        # back to those fields rather than treating this as a hard failure.
        if not content:
            for fallback_attr in ("reasoning_content", "reasoning"):
                fallback_text = getattr(message, fallback_attr, None)
                if fallback_text:
                    logger.warning(
                        "StandardAPIProvider: 'content' was empty; using '%s' "
                        "field instead. Consider raising max_tokens if this "
                        "keeps truncating before a final answer is produced.",
                        fallback_attr,
                    )
                    content = str(fallback_text).strip()
                    break

        # Agent/coding-tuned models sometimes emit a tool_call instead of
        # plain content, even when no tools were declared in the request.
        # If that happened, surface the tool call's arguments as the
        # "content" rather than silently failing, since the model's actual
        # output is sitting there.
        if not content:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                logger.warning(
                    "StandardAPIProvider: 'content' was empty but the model "
                    "returned %d tool_call(s) instead (likely because it's "
                    "an agent/coding-tuned model). Extracting arguments from "
                    "the first tool call as a fallback -- consider switching "
                    "to a plain instruct/chat model for this task instead.",
                    len(tool_calls),
                )
                try:
                    first_call = tool_calls[0]
                    fn = getattr(first_call, "function", None)
                    if fn is not None and getattr(fn, "arguments", None):
                        content = str(fn.arguments).strip()
                except Exception:  # noqa: BLE001
                    pass

        if not content:
            finish_reason = getattr(choice, "finish_reason", "unknown")
            # Dump the raw completion so the *actual* cause is visible
            # (bad/unknown model slug, moderation block, provider routing
            # failure, rate limiting, etc.) instead of guessing blindly.
            try:
                raw_dump = completion.model_dump_json(indent=2)
            except Exception:
                raw_dump = str(completion)
            logger.error(
                "Empty content from model '%s'. Full raw API response for "
                "debugging:\n%s", self.model_name, raw_dump,
            )
            raise LLMProviderError(
                "StandardAPIProvider returned an empty message content "
                f"(finish_reason='{finish_reason}', model='{self.model_name}'). "
                "The raw API response was logged above -- check it for a "
                "moderation flag, routing error, or provider-side issue. "
                "Common causes: (1) the model slug is wrong/does not exist "
                "on this API base -- double check it against the provider's "
                "current model list; (2) a free-tier 'thinking' model spent "
                "its whole max_tokens budget on internal reasoning -- try "
                "raising max_tokens; (3) the specific free model is "
                "temporarily overloaded/deprecated -- try a different model."
            )
        return content


# ---------------------------------------------------------------------------
# Factory: choose a provider via env var / config dict without touching code
# ---------------------------------------------------------------------------

def get_llm_provider(config: Optional[Dict[str, Any]] = None) -> LLMProvider:
    """
    Factory function that builds the configured LLMProvider.

    Resolution order for the provider "kind":
        1. config["provider"], if config is passed and contains it
        2. LLM_PROVIDER environment variable
        3. defaults to "standard"

    Example config dict:
        {
            "provider": "huggingface",
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "api_token": "hf_xxx",
        }

    Or purely via environment variables:
        export LLM_PROVIDER=standard
        export STANDARD_API_KEY=sk-...
        export STANDARD_MODEL=gpt-4o-mini
    """
    config = config or {}
    provider_kind = (config.get("provider") or os.environ.get("LLM_PROVIDER", "standard")).lower()

    if provider_kind in ("huggingface", "hf"):
        return HuggingFaceProvider(
            model_name=config.get("model_name"),
            api_token=config.get("api_token"),
            api_url=config.get("api_url"),
        )
    elif provider_kind in ("standard", "openai", "groq", "gemini"):
        return StandardAPIProvider(
            model_name=config.get("model_name"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
        )
    else:
        raise ValueError(
            f"Unknown LLM provider kind: '{provider_kind}'. "
            "Expected 'huggingface' or 'standard'."
        )


if __name__ == "__main__":
    # Quick manual smoke test (requires real credentials in the environment).
    provider = get_llm_provider()
    reply = provider.generate(
        prompt="Say hello in one short sentence.",
        system="You are a concise assistant.",
        max_tokens=50,
    )
    print(reply)
