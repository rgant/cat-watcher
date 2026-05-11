"""Typed configuration loaded from TOML + environment.

Single source of truth for runtime parameters. Structural config comes from ``config.toml`` (read
via :func:`tomllib.load`); secrets come from environment variables (and optionally a ``.env`` file,
read natively by ``pydantic-settings``).

Resolution order for the TOML path:

1. ``config_path`` argument to :func:`load_config` (CLI ``--config`` flag).
2. ``CAT_WATCHER_CONFIG`` environment variable.
3. ``./config.toml`` (default).

Loaded once at startup; ``ConfigError`` exits the process.
"""

import os
import tomllib
from pathlib import Path
from typing import Annotated, ClassVar, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when config / env is invalid."""


class CameraConfig(BaseModel, extra="forbid"):
    """Per-camera connection parameters (no secrets)."""

    name: str
    display_name: str
    host: str
    port: Annotated[int, Field(ge=1, le=65535)] = 80
    timezone: str | None = None

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            _ = ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            msg = f"invalid IANA timezone: {value!r}"
            raise ValueError(msg) from exc
        return value


class DetectorConfig(BaseModel, extra="forbid"):
    """YOLO detector tuning."""

    model: str = "yolo11n.pt"
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    frames_to_sample: Annotated[int, Field(ge=1)] = 5


class EmailRulesConfig(BaseModel, extra="forbid"):
    """Email alert routing rules (no secrets)."""

    enabled: bool = True
    smtp_host: str = "smtp.gmail.com"
    smtp_port: Annotated[int, Field(ge=1, le=65535)] = 587


class MacOsRulesConfig(BaseModel, extra="forbid"):
    """macOS notification rules."""

    enabled: bool = True


class AlertConfig(BaseModel, extra="forbid"):
    """Alert thresholds + per-channel routing rules."""

    cadence_seconds: Annotated[int, Field(gt=0)] = 900
    inactivity_hours: Annotated[int, Field(gt=0)] = 12
    frequency_window_hours: Annotated[int, Field(gt=0)] = 6
    frequency_threshold_count: Annotated[int, Field(ge=1)] = 8
    cooldown_hours: Annotated[int, Field(gt=0)] = 6
    poller_stuck_minutes: Annotated[int, Field(gt=0)] = 15
    web_down_minutes: Annotated[int, Field(gt=0)] = 5
    alerts_stuck_minutes: Annotated[int, Field(gt=0)] = 30
    web_flapping_window_minutes: Annotated[int, Field(gt=0)] = 30
    web_flapping_threshold_count: Annotated[int, Field(ge=1)] = 5
    backup_stale_hours: Annotated[int, Field(gt=0)] = 36
    disk_low_threshold_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.10
    email: EmailRulesConfig = Field(default_factory=EmailRulesConfig)
    macos: MacOsRulesConfig = Field(default_factory=MacOsRulesConfig)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


class CameraSecrets(BaseSettings):
    """Shared Amcrest camera credentials (one pair for all cameras)."""

    # ``dotenv_filtering="only_existing"`` is required so the dotenv source ignores keys that don't
    # map to a field on this class -- without it, ``extra="forbid"`` rejects sibling-class keys
    # (e.g. ``CAT_WATCHER_GMAIL_USER`` sharing the same .env file) as extra inputs.
    # ``EnvSettingsSource`` already filters by defined fields; this aligns the dotenv source's
    # behavior with that.
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CAT_WATCHER_CAMERA_",
        dotenv_filtering="only_existing",
    )

    username: str
    password: SecretStr


class EmailSecrets(BaseSettings):
    """Outbound email credentials + recipients (env-only)."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CAT_WATCHER_",
        dotenv_filtering="only_existing",
    )

    gmail_user: str
    gmail_app_password: SecretStr
    # ``NoDecode`` keeps EnvSettingsSource from JSON-decoding the raw env string so the
    # mode='before' validator below can split the comma-separated recipients. ``min_length=1``
    # rejects an empty / whitespace-only env value (which would silently swallow alerts).
    alert_to_addresses: Annotated[tuple[str, ...], NoDecode, Field(min_length=1)]

    @field_validator("alert_to_addresses", mode="before")
    @classmethod
    def _split_recipients(cls, value: object) -> object:
        if isinstance(value, str):
            return _split_csv(value)
        return value


class WebAuth(BaseSettings):
    """HTTP basic auth credentials for the web UI (env-only)."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CAT_WATCHER_WEB_",
        dotenv_filtering="only_existing",
    )

    username: str
    password: SecretStr


class WebConfig(BaseModel, extra="forbid"):
    """Web UI structural config (no secrets — see :class:`WebAuth`)."""

    host: str = "0.0.0.0"  # noqa: S104  # binding all interfaces is intentional for LAN access
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    heartbeat_interval_seconds: Annotated[int, Field(gt=0)] = 60
    public_url: str
    display_timezone: str = "America/New_York"


class StorageConfig(BaseModel, extra="forbid"):
    """External-drive wait knobs used by the poller / backup agents."""

    wait_interval_seconds: Annotated[int, Field(gt=0)] = 10
    wait_timeout_seconds: Annotated[int, Field(gt=0)] = 600


class RetentionConfig(BaseModel, extra="forbid"):
    """Row + file retention windows (days)."""

    clip_days: Annotated[int, Field(ge=1)] = 30
    agent_starts_days: Annotated[int, Field(ge=1)] = 30
    alerts_sent_days: Annotated[int, Field(ge=1)] = 30


class BackupConfig(BaseModel, extra="forbid"):
    """SQLite backup cadence + retention."""

    keep_count: Annotated[int, Field(ge=1)] = 7
    cadence_hour: Annotated[int, Field(ge=0, le=23)] = 3
    cadence_minute: Annotated[int, Field(ge=0, le=59)] = 0


class PollerConfig(BaseModel, extra="forbid"):
    """Poller LaunchAgent cadence + cursor advancement guards."""

    cadence_seconds: Annotated[int, Field(gt=0)] = 300
    overlap_minutes: Annotated[int, Field(ge=0)] = 15
    safety_net_hours: Annotated[int, Field(ge=1, le=168)] = 6

    @model_validator(mode="after")
    def _overlap_within_soft_cap(self) -> PollerConfig:
        """Cap ``overlap_minutes`` at 12 ticks of cadence.

        Prevents misconfiguration from growing the per-tick query window unboundedly. Cap =
        ``cadence_seconds * 12 // 60``; with the default cadence (300s), the cap is 60 minutes.
        """
        cap = (self.cadence_seconds * 12) // 60
        if self.overlap_minutes > cap:
            msg = (
                f"overlap_minutes={self.overlap_minutes} exceeds soft cap {cap} for "
                f"cadence_seconds={self.cadence_seconds} (cap = cadence_seconds * 12 / 60); "
                f"lower overlap_minutes or raise cadence_seconds."
            )
            raise ValueError(msg)
        return self


class Config(BaseModel, extra="forbid"):
    """Root configuration model — bound TOML + env at startup."""

    internal_root: Path
    storage_root: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # ``min_length=1`` rejects an empty ``[[cameras]]`` list; ``load_config`` re-wraps the
    # validation error as ``ConfigError``.
    cameras: Annotated[list[CameraConfig], Field(min_length=1)]
    detector: DetectorConfig
    alerts: AlertConfig
    web: WebConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    poller: PollerConfig = Field(default_factory=PollerConfig)

    camera_secrets: CameraSecrets
    email: EmailSecrets
    web_auth: WebAuth

    @model_validator(mode="after")
    def _unique_camera_names(self) -> Config:
        """Reject duplicate camera names so derived storage slugs cannot collide on disk."""
        seen: set[str] = set()
        duplicates: list[str] = []
        for cam in self.cameras:
            if cam.name in seen:
                duplicates.append(cam.name)
            seen.add(cam.name)
        if duplicates:
            joined = ", ".join(sorted(set(duplicates)))
            msg = f"duplicate camera name(s): {joined}"
            raise ValueError(msg)
        return self


def _resolve_config_path(config_path: Path | None) -> Path:
    """Apply the ``arg > env > default`` precedence rule for the TOML location."""
    if config_path is not None:
        return config_path
    env_value = os.environ.get("CAT_WATCHER_CONFIG", "").strip()
    if env_value:
        return Path(env_value)
    return Path("./config.toml")


def _load_toml_dict(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        return cast("dict[str, object]", tomllib.load(fh))


def _load_settings[T: BaseSettings](cls: type[T]) -> T:
    """Construct a ``BaseSettings`` subclass, mapping validation errors to env-var names.

    ``pydantic-settings`` populates fields from env / .env. Missing required fields are re-raised
    as :class:`ConfigError` whose message names the env vars (e.g. ``CAT_WATCHER_CAMERA_PASSWORD``)
    the user must set, rather than leaking the internal ``ClassName.field`` shape from pydantic.
    Other validation failures are also re-raised as :class:`ConfigError` so callers only ever have
    to catch one error type.
    """
    try:
        return cls()
    except ValidationError as exc:
        prefix = cls.model_config.get("env_prefix", "") or ""
        missing = [
            f"{prefix}{err['loc'][0]}".upper()  # dprint-ignore
            for err in exc.errors()
            if err["type"] == "missing" and err["loc"]
        ]
        if missing:
            joined = ", ".join(missing)
            msg = f"missing required env var(s): {joined}"
            raise ConfigError(msg) from exc
        msg = f"invalid env config for {cls.__name__}: {exc}"
        raise ConfigError(msg) from exc


def load_config(config_path: Path | None = None) -> Config:
    """Load and validate config.

    Resolution order: ``config_path`` arg > ``CAT_WATCHER_CONFIG`` env > ``./config.toml``.
    Raises :class:`ConfigError` on any problem.
    """
    resolved = _resolve_config_path(config_path)
    if not resolved.is_file():
        msg = f"config file not found: {resolved}"
        raise ConfigError(msg)

    structural = _load_toml_dict(resolved)

    structural["camera_secrets"] = _load_settings(CameraSecrets)
    structural["email"] = _load_settings(EmailSecrets)
    structural["web_auth"] = _load_settings(WebAuth)

    try:
        return Config.model_validate(structural)
    except ValidationError as exc:
        msg = f"invalid config: {exc}"
        raise ConfigError(msg) from exc
