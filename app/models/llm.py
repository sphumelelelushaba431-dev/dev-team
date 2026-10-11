"""Large language model client.

Supported providers, chosen with LLM_PROVIDER:
  anthropic - Claude via the Anthropic Messages API (ANTHROPIC_API_KEY)
  openai    - OpenAI Chat Completions API (OPENAI_API_KEY)
  groq      - Groq API, free tier, Llama 3.1 (GROQ_API_KEY)
  ollama    - Ollama running locally, free and offline (e.g. llama3.2)
  hf        - a local Hugging Face model (google/flan-t5-base by default)
With "auto", the first available provider is used. If none is available the
client reports itself unavailable and the RAG pipeline falls back to
extractive answers, so the system still runs.
"""
from __future__ import annotations

import logging

import requests

from .. import config

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: str | None = None):
        provider = provider or config.LLM_PROVIDER
        if provider == "auto":
            if config.ANTHROPIC_API_KEY:
                provider = "anthropic"
            elif config.OPENAI_API_KEY:
                provider = "openai"
            elif config.GROQ_API_KEY:
                provider = "groq"
            elif self._ollama_running():
                provider = "ollama"
            elif self._hf_installed():
                provider = "hf"
            else:
                provider = "none"
        self.provider = provider
        self._hf = None

    @staticmethod
    def _ollama_running() -> bool:
        try:
            return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=1).ok
        except Exception:
            return False

    @staticmethod
    def _hf_installed() -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self.provider != "none"

    @property
    def label(self) -> str:
        return {
            "anthropic": f"Anthropic {config.ANTHROPIC_MODEL}",
            "openai": f"OpenAI {config.OPENAI_MODEL}",
            "groq": f"Groq {config.GROQ_MODEL}",
            "ollama": f"Ollama {config.OLLAMA_MODEL}",
            "hf": f"Hugging Face {config.HF_LLM_MODEL}",
        }.get(self.provider, "none")

    def generate(self, system: str, prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
        if self.provider == "anthropic":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": config.ANTHROPIC_MODEL, "max_tokens": max_tokens, "temperature": temperature,
                      "system": system, "messages": [{"role": "user", "content": prompt}]},
                timeout=config.LLM_TIMEOUT)
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip()
        if self.provider in ("openai", "groq", "ollama"):  # OpenAI-compatible chat completions
            url, key, model = {
                "openai": ("https://api.openai.com/v1", config.OPENAI_API_KEY, config.OPENAI_MODEL),
                "groq": ("https://api.groq.com/openai/v1", config.GROQ_API_KEY, config.GROQ_MODEL),
                "ollama": (f"{config.OLLAMA_URL}/v1", "ollama", config.OLLAMA_MODEL),
            }[self.provider]
            r = requests.post(
                f"{url}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                timeout=config.LLM_TIMEOUT if self.provider != "ollama" else max(config.LLM_TIMEOUT, 120))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        if self.provider == "hf":
            if self._hf is None:
                from transformers import pipeline
                self._hf = pipeline("text2text-generation", model=config.HF_LLM_MODEL)
            out = self._hf(f"{system}\n\n{prompt}", max_new_tokens=min(max_tokens, 256), do_sample=False)
            return out[0]["generated_text"].strip()
        raise RuntimeError("No LLM provider configured")
