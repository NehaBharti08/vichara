"""Runtime configuration.

Two layers, deliberately separated:

* :class:`Settings` -- *deployment* concerns that vary by environment and
  include secrets. Sourced from environment variables / ``.env``.
* :class:`PipelineConfig` -- *behavioural* concerns that define what the agent
  actually does: which models, which ceilings, which defences. Sourced from
  ``config/default.yaml`` overlaid with ``config/profiles/<profile>.yaml``.

The split is what makes evaluation reproducible. A profile is a committable,
diffable description of one agent variant, so a results table can cite the
exact configuration that produced it. Secrets must never end up in one.

One deliberate non-behaviour: no API key is required for :class:`Settings` to
validate. Tools degrade to fixture backends when their credentials are absent
(see ``tools/registry.py``), and a config layer that refused to load without a
key would defeat that -- the agent must be able to start with nothing but a
Python interpreter and still be honest about what it cannot do.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class SandboxBackend(enum.StrEnum):
    """Which isolation mechanism executes agent-authored code.

    Both satisfy the same ``Sandbox`` protocol, so calling code never branches
    on this -- only ``sandbox.policy.build_sandbox`` does.
    """

    PYODIDE = "pyodide"
    """CPython compiled to WebAssembly, hosted in a Node subprocess.

    The default everywhere: local development, CI, and the deployed Space.
    Network access is impossible by construction rather than disabled by
    policy, and it needs no daemon -- which is also why the Space can run it
    at all, since Hugging Face offers no Docker-in-Docker.
    """

    DOCKER = "docker"
    """A throwaway container with no network, a read-only root, and rlimits.

    Stronger isolation and arbitrary pip packages, but it needs a daemon. Kept
    fully implemented and exercised in CI; see docs/THREAT_MODEL.md for what
    each backend does and does not stop.
    """


class Settings(BaseSettings):
    """Environment-sourced configuration. Contains secrets; never serialise it."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Model providers ----------------------------------------------------
    google_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="GOOGLE_API_KEY")
    """Gemini API key. Absent is a valid state: the agent starts, reports the
    model provider as unavailable, and every LLM-dependent path fails with a
    single legible error instead of a stack trace per call site."""

    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    """Kept implemented and tested but unused by default. The free tier is the
    binding constraint on evaluation throughput; this is the escape hatch when
    a full sweep is needed in twenty minutes rather than two days."""

    # -- Tool credentials ---------------------------------------------------
    tavily_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="TAVILY_API_KEY")
    """Web search. Absent -> the search tool falls back to its fixture backend."""

    vidyarag_url: str | None = Field(default=None, validation_alias="VIDYARAG_URL")
    """Base URL of the VidyaRAG retrieval service. Absent -> the retrieval tool
    falls back to the committed fixture corpus. The agent must never hard-depend
    on a sibling service being deployed."""

    # -- Behaviour selection ------------------------------------------------
    profile: str = Field(default="baseline", validation_alias="VICHARA_PROFILE")
    sandbox_backend: SandboxBackend | None = Field(default=None, validation_alias="SANDBOX_BACKEND")
    """Overrides the profile's sandbox choice. Exists so CI can force the
    Docker backend through the same protocol tests without a separate profile."""

    # -- Paths --------------------------------------------------------------
    workspace_root: str = Field(default="sessions", validation_alias="WORKSPACE_ROOT")
    """Parent of every per-session workspace. The file tool refuses to resolve
    outside the session directory beneath this; see tools/files/workspace.py."""

    cache_path: str = Field(default="data/cache/llm_cache.sqlite", validation_alias="CACHE_PATH")
    checkpoint_path: str = Field(
        default="data/cache/checkpoints.sqlite", validation_alias="CHECKPOINT_PATH"
    )

    # -- Observability ------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(default="json", validation_alias="LOG_FORMAT")

    @model_validator(mode="after")
    def _check_url_shape(self) -> Settings:
        """Fail at startup rather than at first retrieval call."""
        if self.vidyarag_url is not None:
            url = self.vidyarag_url.strip()
            if not url:
                self.vidyarag_url = None
            elif not url.startswith(("http://", "https://")):
                raise ValueError(f"VIDYARAG_URL must start with http:// or https://, got {url!r}")
        return self

    def resolved(self, value: str) -> Path:
        """Interpret a configured path relative to the repository root."""
        path = Path(value)
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key.get_secret_value())

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key.get_secret_value())


# ---------------------------------------------------------------------------
# Behavioural configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """One model per role.

    Roles are separated because they have genuinely different cost profiles.
    ``compress`` and ``judge`` run on the cheapest model available; ``planner``
    is the only one a stronger model is a candidate for, and the ablation in
    docs/EVALUATION.md exists to decide whether it earns the quota.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["google", "openai"] = "google"
    agent: str = "gemini-3.5-flash-lite"
    planner: str = "gemini-3.5-flash-lite"
    judge: str = "gemini-3.5-flash"
    compress: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    max_output_tokens: int = 2048


class BudgetConfig(BaseModel):
    """Hard ceilings on a single task.

    On a free tier the scarce resource is requests per day, not dollars, so
    ``max_llm_requests`` is the ceiling that actually fires. ``max_est_usd`` is
    enforced too and will simply never trip until the provider is switched --
    it exists so the guardrail is real on the day it matters.

    ``max_steps`` is provisional until the gold optimal paths are annotated in
    Phase 4; the method is to set it at roughly 2.5x the worst-case optimal
    path, which leaves headroom for two full failure-reflect-retry cycles.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps: int = 12
    max_llm_requests: int = 25
    max_tokens: int = 150_000
    max_wall_clock_s: float = 120.0
    max_est_usd: float = 0.05
    max_plan_revisions: int = 3


class ToolLimitsConfig(BaseModel):
    """Per-tool call ceilings.

    A fourth reformulation of a query that has already failed three times has
    near-zero marginal yield, so the cap is not just a safety rail: it forces
    the agent down the reflect path instead of letting it grind.
    """

    model_config = ConfigDict(extra="forbid")

    default: int = 4
    per_tool: dict[str, int] = Field(
        default_factory=lambda: {
            "textbook_search": 4,
            "web_search": 4,
            "run_python": 2,
            "workspace_file": 3,
        }
    )

    def limit_for(self, tool_name: str) -> int:
        return self.per_tool.get(tool_name, self.default)


class LoopConfig(BaseModel):
    """Repeated-action detection.

    The identical-fingerprint threshold is 2, not 3. One exact repeat of the
    same tool with the same arguments is already a bug -- the observation did
    not change, so neither will the result.
    """

    model_config = ConfigDict(extra="forbid")

    identical_threshold: int = 2
    near_repeat_window: int = 3
    near_repeat_similarity: float = 0.9


class MemoryConfig(BaseModel):
    """Tiered retention.

    Summarisation here is not about context overflow -- Gemini Flash holds a
    million tokens and a twelve-step trajectory never approaches it. It is
    about cost, which is quadratic because every step resends the whole
    trajectory, and about attention dilution, which measurably degrades tool
    selection. Both are testable claims; see docs/EVALUATION.md.
    """

    model_config = ConfigDict(extra="forbid")

    verbatim_recent_observations: int = 3
    externalize_over_bytes: int = 2048
    soft_limit_tokens: int = 12_000
    compress_every_n_steps: int = 5
    preserve_provenance_tags: bool = True
    """Non-negotiable in any profile that claims injection defence. A summary
    that drops the 'this came from an untrusted document' marker re-emits the
    payload as trusted assistant narration -- summarisation becomes a
    laundering channel. There is a dedicated test for this."""


class PlanningConfig(BaseModel):
    """Advisory plans, not binding ones.

    The plan is a typed artifact rather than prose in a scratchpad so that
    Phase 4 can compare three separate objects: the annotated optimal path,
    the agent's plan, and the agent's actual trajectory. Deviations are logged
    rather than prevented -- a plan the agent cannot abandon is worse than no
    plan when a tool fails.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_plan_steps: int = 1
    max_plan_steps: int = 6
    advisory: bool = True


class ReflectionConfig(BaseModel):
    """When to stop and think.

    Reflecting on every step roughly doubles request count for marginal gain,
    and requests are the binding budget. These triggers concentrate it where
    it changes behaviour.
    """

    model_config = ConfigDict(extra="forbid")

    on_tool_error: bool = True
    on_low_information: bool = True
    on_plan_step_complete: bool = True
    every_n_steps: int = 3
    skip_first_step: bool = True


class SandboxConfig(BaseModel):
    """Resource ceilings for agent-authored code.

    Generous enough that a numpy computation completes, tight enough that
    abuse is bounded. What these limits do *not* stop is the subject of
    docs/THREAT_MODEL.md, which is the more useful half of the document.
    """

    model_config = ConfigDict(extra="forbid")

    backend: SandboxBackend = SandboxBackend.PYODIDE
    cpu_seconds: float = 5.0
    memory_mb: int = 256
    wall_clock_s: float = 10.0
    max_output_bytes: int = 65_536
    network: bool = False
    warm_worker: bool = True
    """Amortises Pyodide's 1-2s cold start across a session. Without it, a
    sandbox call costs more in latency than the computation saves -- which is
    the only argument that was ever available for shipping a calculator tool."""


class InjectionConfig(BaseModel):
    """Defences against instructions arriving through tool output.

    Off in ``baseline`` on purpose. The baseline attack success rate is a
    published number, and it cannot be measured against a configuration that
    already defends. See docs/PROMPT_INJECTION.md.
    """

    model_config = ConfigDict(extra="forbid")

    tag_provenance: bool = True
    """Kept on even in baseline: it is instrumentation, not a defence. Without
    span tagging there is no way to attribute a compromised trajectory."""

    neutralise_delimiters: bool = False
    strip_imperatives: bool = False
    detector_enabled: bool = False
    on_detection: Literal["log", "strip", "quarantine", "abort"] = "log"


class ToolsConfig(BaseModel):
    """Retrieval and search shaping. Registration itself lives in tools.yaml."""

    model_config = ConfigDict(extra="forbid")

    retrieval_top_k: int = 5
    search_max_results: int = 5
    tool_timeout_s: float = 30.0
    tool_max_retries: int = 2
    max_output_bytes: int = 16_384
    """Ceiling on what one tool call may put into the context. Five retrieved
    passages plus five search results already approach this; without a cap a
    single call can consume more of the step budget than the reasoning does."""


class PipelineConfig(BaseModel):
    """A complete, reproducible description of one agent variant."""

    model_config = ConfigDict(extra="forbid")

    name: str = "baseline"
    description: str = ""
    models: ModelConfig = Field(default_factory=ModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    tool_limits: ToolLimitsConfig = Field(default_factory=ToolLimitsConfig)
    loops: LoopConfig = Field(default_factory=LoopConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    injection: InjectionConfig = Field(default_factory=InjectionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_pipeline_config(profile: str, config_dir: Path | None = None) -> PipelineConfig:
    """Load ``default.yaml`` overlaid with ``profiles/<profile>.yaml``.

    ``extra="forbid"`` on every model means a typo in a profile key is a
    startup error rather than a silently ignored setting -- which would
    otherwise invalidate a benchmark without anyone noticing.
    """
    directory = config_dir or CONFIG_DIR
    base = _read_yaml(directory / "default.yaml")
    profile_path = directory / "profiles" / f"{profile}.yaml"
    if not profile_path.exists():
        available = sorted(p.stem for p in (directory / "profiles").glob("*.yaml"))
        raise FileNotFoundError(
            f"Unknown profile {profile!r}. Available: {', '.join(available) or '(none)'}"
        )
    merged = _deep_merge(base, _read_yaml(profile_path))
    merged.setdefault("name", profile)
    return PipelineConfig.model_validate(merged)
