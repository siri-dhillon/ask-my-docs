"""Pluggable LLM providers behind one tiny interface.

Every provider returns *schema-validated* objects (structured outputs), so downstream code
never regex-parses model text. Selected by ASKDOCS_LLM_PROVIDER: anthropic | openai | ollama | fake.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from askdocs.config import Settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens


class LLMError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str
    model: str

    def structured(
        self, system: str, user: str, schema: type[T], max_tokens: int = 2000, cache_system: bool = False
    ) -> tuple[T, Usage]: ...

    def text(self, system: str, user: str, max_tokens: int = 500, cache_system: bool = False) -> tuple[str, Usage]: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY (or an `ant auth login` profile)

    def _system(self, system: str, cache: bool) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:  # e.g. contextual retrieval: the same document is reused for every chunk
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _usage(resp: Any) -> Usage:
        u = resp.usage
        return Usage(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0)

    @staticmethod
    def _check(resp: Any) -> None:
        if resp.stop_reason == "refusal":
            raise LLMError("model refused the request")
        if resp.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")

    def structured(self, system, user, schema, max_tokens=2000, cache_system=False):
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system(system, cache_system),
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": "low"},  # grounded extraction-style task: low effort keeps latency down
        )
        self._check(resp)
        if resp.parsed_output is None:
            raise LLMError("no parsed output")
        return resp.parsed_output, self._usage(resp)

    def text(self, system, user, max_tokens=500, cache_system=False):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system(system, cache_system),
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
        )
        self._check(resp)
        out = "".join(b.text for b in resp.content if b.type == "text").strip()
        return out, self._usage(resp)


class OpenAICompatibleProvider:
    """OpenAI, or anything speaking its API (Ollama exposes /v1 on localhost)."""

    def __init__(self, model: str, name: str = "openai", base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.name = name
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI()

    @staticmethod
    def _usage(resp: Any) -> Usage:
        u = resp.usage
        return Usage(u.prompt_tokens, u.completion_tokens) if u else Usage()

    def structured(self, system, user, schema, max_tokens=2000, cache_system=False):
        resp = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=schema,
            max_completion_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        if msg.parsed is None:
            raise LLMError(f"no parsed output (refusal={getattr(msg, 'refusal', None)})")
        return msg.parsed, self._usage(resp)

    def text(self, system, user, max_tokens=500, cache_system=False):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip(), self._usage(resp)


@dataclass
class FakeProvider:
    """Deterministic offline provider for tests and CI-without-secrets.

    Register a handler per output schema: handler(system, user) -> schema instance.
    Handlers for the app's own schemas are installed by `askdocs.generate.fake_handlers`.
    """

    name: str = "fake"
    model: str = "fake-1"
    handlers: dict[type, Callable[[str, str], BaseModel]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def structured(self, system, user, schema, max_tokens=2000, cache_system=False):
        self.calls.append((schema.__name__, user))
        if schema not in self.handlers:
            raise LLMError(f"FakeProvider has no handler for {schema.__name__}")
        out = self.handlers[schema](system, user)
        return out, Usage(len(user) // 4, len(json.dumps(out.model_dump())) // 4)

    def text(self, system, user, max_tokens=500, cache_system=False):
        self.calls.append(("text", user))
        first = re.split(r"(?<=[.!?])\s", user.strip(), maxsplit=1)[0]
        return first[:200], Usage(len(user) // 4, 20)


def build_provider(settings: Settings, model: str | None = None) -> LLMProvider:
    model = model or settings.llm_model
    match settings.llm_provider:
        case "anthropic":
            return AnthropicProvider(model)
        case "openai":
            return OpenAICompatibleProvider(model)
        case "ollama":
            return OpenAICompatibleProvider(model, name="ollama", base_url=settings.ollama_base_url, api_key="ollama")
        case "fake":
            from askdocs.generate.fake_handlers import default_fake

            return default_fake()
    raise ValueError(f"unknown provider {settings.llm_provider}")
