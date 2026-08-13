"""The one place a vendor SDK is imported.

Everything else in the project asks for a *role* -- ``agent``, ``planner``,
``judge``, ``compress`` -- and gets back something it can call. Switching
provider is then a config change rather than a refactor, and the planner
ablation in Phase 4 is a one-line profile edit rather than a code branch.

Every call goes through the same path: cache lookup, rate-limited request,
accounting. That ordering matters. A cache hit costs no quota and no time, so
it must be checked before the bucket; and accounting must record the hit as a
call that happened, or the eval's request counts silently understate what a
cold run would cost.

## The content-blocks trap

langchain-core 1.x returns ``AIMessage.content`` as a **list of content
blocks**, not a string:

    [{'type': 'text', 'text': '17 * 23 = 391', 'extras': {...}}]

Anything doing ``str(response.content)`` gets that literal repr in its answer.
:func:`text_of` is the only supported way to read a response as text, and it
exists because the Phase 1 spike caught this before the agent was written.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from vichara.llm.accounting import (
    CallRecord,
    Ledger,
    estimate_usd,
    usage_from_response,
)
from vichara.llm.cache import ResponseCache, cache_key
from vichara.llm.ratelimit import TokenBucket, call_with_backoff
from vichara.logging import get_logger
from vichara.settings import PipelineConfig, Settings

log = get_logger(__name__)

Role = Literal["agent", "planner", "judge", "compress"]


class ProviderUnavailable(RuntimeError):
    """No usable credential for the configured provider.

    Raised once, at construction, with a sentence that says what to do --
    rather than surfacing as an authentication error from inside a graph node
    where it reads as an agent failure.
    """


def text_of(response: Any) -> str:
    """Read a model response as plain text.

    Handles both the string form and langchain-core 1.x content blocks. Use
    this everywhere; ``str(response.content)`` is a bug.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _message_to_cacheable(message: BaseMessage) -> dict[str, Any]:
    """A stable dict for hashing. Only the parts that change the answer."""
    return {
        "type": message.type,
        "content": message.content,
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }


def _serialise(message: AIMessage) -> dict[str, Any]:
    """Store an explicit payload rather than LangChain's own serialiser.

    The internal format has changed across minor versions, and a cache that
    cannot be read after an upgrade is worse than no cache -- it fails at the
    moment you most want the old numbers to still reproduce.
    """
    return {
        "content": message.content,
        "tool_calls": [
            {
                "name": call["name"],
                "args": call["args"],
                "id": call.get("id"),
                "type": "tool_call",
            }
            for call in (message.tool_calls or [])
        ],
        "usage_metadata": message.usage_metadata,
    }


def _deserialise(payload: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content=payload.get("content", ""),
        tool_calls=payload.get("tool_calls") or [],
        usage_metadata=payload.get("usage_metadata"),
    )


class ModelClient:
    """A model bound to one role, with caching, pacing and accounting."""

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        role: Role,
        cache: ResponseCache,
        bucket: TokenBucket,
        ledger: Ledger,
        temperature: float,
        seed: int | None = None,
    ) -> None:
        self._model = model
        self.model_name = model_name
        self.role = role
        self._cache = cache
        self._bucket = bucket
        self._ledger = ledger
        self.temperature = temperature
        self.seed = seed

    @property
    def deterministic(self) -> bool:
        """Only deterministic calls may be cached.

        At temperature > 0 repeated calls are *supposed* to differ, and caching
        would collapse the cross-seed variance Phase 4 reports as a result.
        """
        return self.temperature == 0.0

    def invoke(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[Any] | None = None,
    ) -> AIMessage:
        """One call, with tools optionally bound."""
        tool_names = sorted(getattr(t, "name", str(t)) for t in (tools or []))
        key = cache_key(
            model=self.model_name,
            messages=[_message_to_cacheable(m) for m in messages],
            tools=tool_names,
            temperature=self.temperature,
            seed=self.seed,
        )

        if self.deterministic:
            cached = self._cache.get(key)
            if cached is not None:
                message = _deserialise(cached)
                inputs, outputs, cached_tokens = usage_from_response(message)
                self._ledger.record(
                    CallRecord(
                        model=self.model_name,
                        role=self.role,
                        input_tokens=inputs,
                        output_tokens=outputs,
                        cached_tokens=cached_tokens,
                        cache_hit=True,
                    )
                )
                log.debug("cache hit", role=self.role, key=key[:12])
                return message

        runnable = self._model.bind_tools(tools) if tools else self._model
        started = time.perf_counter()
        try:
            response = call_with_backoff(lambda: runnable.invoke(messages), bucket=self._bucket)
        except Exception as exc:
            self._ledger.record(
                CallRecord(
                    model=self.model_name,
                    role=self.role,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        inputs, outputs, cached_tokens = usage_from_response(response)
        self._ledger.record(
            CallRecord(
                model=self.model_name,
                role=self.role,
                input_tokens=inputs,
                output_tokens=outputs,
                cached_tokens=cached_tokens,
                latency_ms=latency_ms,
                est_usd=estimate_usd(self.model_name, inputs, outputs),
            )
        )
        self._cache.record_request(time.strftime("%Y-%m-%d"))

        if self.deterministic and isinstance(response, AIMessage):
            self._cache.put(
                key, model=self.model_name, role=self.role, payload=_serialise(response)
            )

        return response  # type: ignore[no-any-return]

    def structured(self, messages: list[BaseMessage], schema: type[BaseModel]) -> Any:
        """Call with a typed output schema. Used by the planner and reflector."""
        key = cache_key(
            model=self.model_name,
            messages=[_message_to_cacheable(m) for m in messages],
            tools=[],
            temperature=self.temperature,
            seed=self.seed,
            structured=schema.__name__,
        )

        if self.deterministic:
            cached = self._cache.get(key)
            if cached is not None:
                self._ledger.record(
                    CallRecord(model=self.model_name, role=self.role, cache_hit=True)
                )
                return schema.model_validate(cached["structured"])

        started = time.perf_counter()
        runnable = self._model.with_structured_output(schema)
        result = call_with_backoff(lambda: runnable.invoke(messages), bucket=self._bucket)
        self._ledger.record(
            CallRecord(
                model=self.model_name,
                role=self.role,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        self._cache.record_request(time.strftime("%Y-%m-%d"))

        if self.deterministic and isinstance(result, BaseModel):
            self._cache.put(
                key,
                model=self.model_name,
                role=self.role,
                payload={"structured": result.model_dump(mode="json")},
            )
        return result


class Provider:
    """Builds role-bound clients that share a cache, bucket and ledger."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig,
        *,
        ledger: Ledger | None = None,
        cache: ResponseCache | None = None,
        seed: int | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.ledger = ledger or Ledger()
        self.cache = cache or ResponseCache(settings.resolved(settings.cache_path))
        # One bucket for every role: the provider's quota is per key, not per
        # role, so separate buckets would let four callers each pace correctly
        # and still exceed the limit together.
        self.bucket = TokenBucket()
        self.seed = seed
        self._models: dict[str, Any] = {}

    def _model_name(self, role: Role) -> str:
        return {
            "agent": self.config.models.agent,
            "planner": self.config.models.planner,
            "judge": self.config.models.judge,
            "compress": self.config.models.compress,
        }[role]

    def _build(self, model_name: str) -> Any:
        """Instantiate a vendor model. Cached per name within this provider."""
        if model_name in self._models:
            return self._models[model_name]

        provider = self.config.models.provider
        if provider == "google":
            if not self.settings.has_google_key:
                raise ProviderUnavailable(
                    "GOOGLE_API_KEY is not set. Add it to .env -- a free key comes "
                    "from https://aistudio.google.com/apikey. Tools and the sandbox "
                    "work without one; the agent loop does not."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.config.models.temperature,
                max_output_tokens=self.config.models.max_output_tokens,
                google_api_key=self.settings.google_api_key.get_secret_value(),
            )
        else:
            if not self.settings.openai_api_key.get_secret_value():
                raise ProviderUnavailable("OPENAI_API_KEY is not set.")
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=model_name,
                temperature=self.config.models.temperature,
                max_tokens=self.config.models.max_output_tokens,
                api_key=self.settings.openai_api_key.get_secret_value(),
            )

        self._models[model_name] = model
        return model

    def get(self, role: Role) -> ModelClient:
        model_name = self._model_name(role)
        return ModelClient(
            self._build(model_name),
            model_name=model_name,
            role=role,
            cache=self.cache,
            bucket=self.bucket,
            ledger=self.ledger,
            temperature=self.config.models.temperature,
            seed=self.seed,
        )

    def available(self) -> tuple[bool, str]:
        """Whether the agent loop can run at all. Never raises."""
        try:
            self._build(self._model_name("agent"))
        except ProviderUnavailable as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 - a missing SDK is a config problem, not a crash
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"{self.config.models.provider}, agent={self.config.models.agent}"
