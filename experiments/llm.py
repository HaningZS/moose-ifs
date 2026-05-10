"""LLM backend abstraction for experiment variants.

Provides a Protocol-based interface so variants work with any LLM backend
(OpenAI, Anthropic, local). MockLLM for testing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol


class LLMBackend(Protocol):
    """Protocol for LLM backends used by experiment variants."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        """Generate a completion from the LLM."""
        ...


@dataclass
class MockLLM:
    """Deterministic mock LLM for testing. Cycles through responses."""

    responses: list[str]
    call_log: list[dict] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        self.call_log.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
        })
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response



def extract_code(response: str) -> str:
    """Strip markdown fences if present, return raw .i content.

    Handles ```...```, ```ini...```, ```hit...``` etc.
    If no fences found, returns the response stripped of leading/trailing whitespace.
    """
    match = re.search(r"```\w*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


@dataclass
class AnthropicLLM:
    """Anthropic Claude backend."""

    model: str = "claude-sonnet-4-6"
    api_key: str | None = None

    def __post_init__(self):
        import anthropic

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key or set the environment variable."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        response = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=4096,
        )
        return extract_code(response.content[0].text)


@dataclass
class GeminiLLM:
    """Google Gemini backend via google-genai SDK.

    Supports two modes:
    - API key: set GEMINI_API_KEY env var (AI Studio)
    - GCP Vertex: set GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION env vars
      (uses application default credentials via `gcloud auth`)
    """

    model: str = "gemini-2.5-flash"
    api_key: str | None = None
    project: str | None = None
    location: str | None = None

    def __post_init__(self):
        from google import genai

        self.project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = self.location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        key = self.api_key or os.environ.get("GEMINI_API_KEY")

        if self.project:
            # GCP Vertex AI mode
            self._client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        elif key:
            # API key mode (AI Studio)
            self._client = genai.Client(api_key=key)
        else:
            raise ValueError(
                "Set GEMINI_API_KEY (AI Studio) or GOOGLE_CLOUD_PROJECT (Vertex AI)."
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        return extract_code(response.text)


@dataclass
class OpenAICompatibleLLM:
    """OpenAI-compatible backend (covers OpenAI API, vLLM, TGI).

    Set use_completion_tokens=True for models that require max_completion_tokens
    instead of max_tokens (e.g. gpt-5.x, o1, o3, o4 series).
    """

    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    use_completion_tokens: bool = False

    def __post_init__(self):
        import openai

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set. Pass api_key or set the environment variable."
            )
        self._client = openai.OpenAI(api_key=key, base_url=self.base_url)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> str:
        token_kwarg = (
            {"max_completion_tokens": 4096}
            if self.use_completion_tokens
            else {"max_tokens": 4096}
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            **token_kwarg,
        )
        return extract_code(response.choices[0].message.content)
