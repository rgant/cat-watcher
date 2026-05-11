"""Tests for cat_watcher.config."""

import textwrap
from pathlib import Path

import pytest

from cat_watcher.config import ConfigError, load_config

_VALID_TOML = textwrap.dedent("""\
    internal_root = "./data"
    storage_root = "./data"
    log_level = "INFO"

    [[cameras]]
    name = "pantry"
    display_name = "Pantry Litter Box Camera"
    host = "10.0.0.1"
    port = 80

    [[cameras]]
    name = "office"
    display_name = "Office Litter Box Camera"
    host = "10.0.0.2"
    port = 80

    [detector]
    model = "yolo11n.pt"
    confidence_threshold = 0.35
    frames_to_sample = 5

    [alerts]
    inactivity_hours = 12
    frequency_window_hours = 6
    frequency_threshold_count = 8
    cooldown_hours = 6

    [alerts.email]
    enabled = true
    smtp_host = "smtp.gmail.com"
    smtp_port = 587

    [alerts.macos]
    enabled = true

    [web]
    host = "0.0.0.0"
    port = 8000
    heartbeat_interval_seconds = 60
    public_url = "http://mac-mini.local:8000"
    display_timezone = "America/New_York"
""")


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the env vars BaseSettings reads for shared camera, email, and web auth secrets."""
    monkeypatch.setenv("CAT_WATCHER_CAMERA_USERNAME", "shared-user")
    monkeypatch.setenv("CAT_WATCHER_CAMERA_PASSWORD", "shared-pass")
    monkeypatch.setenv("CAT_WATCHER_GMAIL_USER", "alerts@example.com")
    monkeypatch.setenv("CAT_WATCHER_GMAIL_APP_PASSWORD", "app-pw")
    monkeypatch.setenv("CAT_WATCHER_ALERT_TO_ADDRESSES", "me@example.com")
    monkeypatch.setenv("CAT_WATCHER_WEB_USERNAME", "admin")
    monkeypatch.setenv("CAT_WATCHER_WEB_PASSWORD", "secret")


def test_load_config_reads_toml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful load merges TOML structure with env-injected camera + email + web secrets."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    cfg = load_config()

    assert cfg.log_level == "INFO"
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].name == "pantry"
    assert cfg.camera_secrets.username == "shared-user"
    assert cfg.camera_secrets.password.get_secret_value() == "shared-pass"
    assert cfg.detector.confidence_threshold == 0.35
    assert cfg.alerts.frequency_threshold_count == 8
    assert cfg.web.public_url == "http://mac-mini.local:8000"
    assert cfg.web_auth.username == "admin"
    assert cfg.web_auth.password.get_secret_value() == "secret"
    assert cfg.email.gmail_user == "alerts@example.com"
    assert cfg.email.alert_to_addresses == ("me@example.com",)


def test_missing_camera_password_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing shared camera password env var raises ``ConfigError`` naming the env var."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)
    monkeypatch.delenv("CAT_WATCHER_CAMERA_PASSWORD")

    # The user-facing contract: the error message names the env var the user must set, not the
    # internal pydantic-settings field path.
    with pytest.raises(ConfigError, match=r"CAT_WATCHER_CAMERA_PASSWORD"):
        _ = load_config()


def test_empty_alert_to_addresses_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``CAT_WATCHER_ALERT_TO_ADDRESSES`` env var raises ``ConfigError``."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)
    monkeypatch.setenv("CAT_WATCHER_ALERT_TO_ADDRESSES", "")

    with pytest.raises(ConfigError, match=r"alert_to_addresses"):
        _ = load_config()


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("a@b", ("a@b",)),
        ("  a@b  ,  c@d  ", ("a@b", "c@d")),
        ("a@b,", ("a@b",)),
        ("a@b,,c@d", ("a@b", "c@d")),
    ],
)
def test_alert_to_addresses_csv_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: tuple[str, ...],
) -> None:
    """``CAT_WATCHER_ALERT_TO_ADDRESSES`` parses CSV with whitespace + empty-field tolerance."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)
    monkeypatch.setenv("CAT_WATCHER_ALERT_TO_ADDRESSES", env_value)

    cfg = load_config()

    assert cfg.email.alert_to_addresses == expected


def test_unknown_field_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unmapped top-level TOML field is rejected by ``extra='forbid'`` as ``ConfigError``."""
    bad = _VALID_TOML + "\nbogus_field = 42\n"
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(bad)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError):
        _ = load_config()


def test_load_config_path_argument_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``config_path`` arg wins over ``CAT_WATCHER_CONFIG`` env var."""
    # Env var points at a non-existent path: would raise if the loader read it.
    bogus_env_path = tmp_path / "env_only.toml"
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(bogus_env_path))
    arg_path = tmp_path / "arg.toml"
    _ = arg_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    cfg = load_config(config_path=arg_path)

    assert cfg.web.public_url == "http://mac-mini.local:8000"


def test_camera_timezone_invalid_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid IANA zone surfaces as ``ConfigError`` (not ``ValidationError``)."""
    toml = _VALID_TOML.replace(
        '[[cameras]]\nname = "pantry"\ndisplay_name = "Pantry Litter Box Camera"\nhost = "10.0.0.1"\nport = 80\n',
        '[[cameras]]\nname = "pantry"\ndisplay_name = "Pantry Litter Box Camera"\nhost = "10.0.0.1"\nport = 80\ntimezone = "Not/Real"\n',
    )
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(toml)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError, match="Not/Real"):
        _ = load_config()


def _toml_with_field(section: str, field: str, value: float) -> str:
    """Return ``_VALID_TOML`` with ``[section].field = value`` spliced in.

    Splice in place if the field already exists; under the section header if the section exists but
    the field doesn't; or as a fresh trailing section. Avoids TOML duplicate-key / duplicate-section
    errors in all three cases.
    """
    existing_line_prefix = f"{field} = "
    section_header = f"[{section}]\n"
    if existing_line_prefix in _VALID_TOML:
        lines = _VALID_TOML.splitlines(keepends=True)
        new_lines = [f"{field} = {value}\n" if line.startswith(existing_line_prefix) else line for line in lines]
        return "".join(new_lines)
    if section_header in _VALID_TOML:
        return _VALID_TOML.replace(section_header, f"{section_header}{field} = {value}\n")
    return _VALID_TOML + f"\n{section_header}{field} = {value}\n"


@pytest.mark.parametrize(
    ("spec", "should_raise"),
    [
        # BackupConfig.cadence_hour bounds (ge=0, le=23) -- wraparound 24 must fail.
        (("backup", "cadence_hour", 24), True),
        (("backup", "cadence_hour", 23), False),
        # BackupConfig.cadence_minute bounds (ge=0, le=59) -- wraparound 60 must fail.
        (("backup", "cadence_minute", 60), True),
        (("backup", "cadence_minute", 59), False),
        # BackupConfig.keep_count (ge=1) -- 0 keeps nothing.
        (("backup", "keep_count", 0), True),
        # PollerConfig.cadence_seconds (gt=0) -- 0 is a tight loop.
        (("poller", "cadence_seconds", 0), True),
        # Lowest cadence compatible with the default overlap_minutes=15 under the soft cap
        # (overlap_minutes <= (cadence_seconds * 12) // 60, i.e. ceil(15 * 60 / 12) = 75).
        (("poller", "cadence_seconds", 75), False),
        # PollerConfig.overlap_minutes (ge=0) -- negative rejected; 0 is the inclusive floor.
        (("poller", "overlap_minutes", -1), True),
        (("poller", "overlap_minutes", 0), False),
        # PollerConfig.safety_net_hours (ge=1, le=168) -- one week ceiling; 0 would fire every tick.
        (("poller", "safety_net_hours", 0), True),
        (("poller", "safety_net_hours", 1), False),
        (("poller", "safety_net_hours", 168), False),
        (("poller", "safety_net_hours", 200), True),
        # AlertConfig.cadence_seconds (gt=0).
        (("alerts", "cadence_seconds", 0), True),
        # One representative _count field (ge=1).
        (("alerts", "frequency_threshold_count", 0), True),
        # One representative _minutes field (gt=0).
        (("alerts", "poller_stuck_minutes", 0), True),
        # CameraConfig.port + WebConfig.port (ge=1, le=65535).
        # `port = ` appears in both [[cameras]] and [web]; helper rewrites all.
        # Either model raising ConfigError satisfies the test.
        (("web", "port", 0), True),
        (("web", "port", 65536), True),
        (("web", "port", 65535), False),
        # EmailRulesConfig.smtp_port (ge=1, le=65535) -- only in [alerts.email].
        (("alerts.email", "smtp_port", 0), True),
        (("alerts.email", "smtp_port", 65536), True),
        # WebConfig.heartbeat_interval_seconds (gt=0) -- 0 is a tight loop.
        (("web", "heartbeat_interval_seconds", 0), True),
        # DetectorConfig.confidence_threshold (ge=0.0, le=1.0) -- both endpoints inclusive.
        (("detector", "confidence_threshold", 1.5), True),
        (("detector", "confidence_threshold", -0.1), True),
        (("detector", "confidence_threshold", 0.0), False),
        (("detector", "confidence_threshold", 1.0), False),
        # DetectorConfig.frames_to_sample (ge=1) -- 0 frames means no detection.
        (("detector", "frames_to_sample", 0), True),
        (("detector", "frames_to_sample", 1), False),
        # AlertConfig.disk_low_threshold_fraction (gt=0, lt=1) -- both endpoints + above-1 fail.
        (("alerts", "disk_low_threshold_fraction", 0.0), True),
        (("alerts", "disk_low_threshold_fraction", 1.0), True),
        (("alerts", "disk_low_threshold_fraction", 1.5), True),
    ],
)
def test_field_bound_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: tuple[str, str, float],
    *,
    should_raise: bool,
) -> None:
    """Numeric fields enforce ``Field``-declared bounds that prevent LaunchAgent footguns."""
    section, field, value = spec
    body = _toml_with_field(section, field, value)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    if should_raise:
        with pytest.raises(ConfigError):
            _ = load_config()
    else:
        _ = load_config()  # must not raise


def test_whitespace_only_env_var_falls_through_to_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only ``CAT_WATCHER_CONFIG`` is treated as unset."""
    default_path = tmp_path / "config.toml"
    _ = default_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", "   ")
    _set_env(monkeypatch)

    cfg = load_config()  # no config_path arg -> must fall through env -> default

    assert cfg.web.public_url == "http://mac-mini.local:8000"


def test_example_config_loads_against_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``config.example.toml`` must remain a valid ``Config`` (catches schema drift)."""
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "config.example.toml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(example))
    _set_env(monkeypatch)

    cfg = load_config()

    # Sanity checks tied to the example file's hardcoded values.
    assert cfg.poller.cadence_seconds == 300
    assert cfg.alerts.cadence_seconds == 900
    assert cfg.backup.keep_count == 7
    assert cfg.web.display_timezone == "America/New_York"


def test_missing_config_file_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ConfigError`` (not a raw ``OSError``) when the resolved TOML path doesn't exist."""
    missing = tmp_path / "does-not-exist.toml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(missing))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError, match="config file not found"):
        _ = load_config()


def test_secret_str_masks_passwords_in_repr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``SecretStr`` fields render as ``'**********'`` in str/repr to prevent log leakage."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    cfg = load_config()

    assert str(cfg.camera_secrets.password) == "**********"
    assert str(cfg.email.gmail_app_password) == "**********"
    assert str(cfg.web_auth.password) == "**********"
    # The mask MUST also appear in repr, so accidental logging via f-string / %r is safe.
    assert "shared-pass" not in repr(cfg.camera_secrets.password)
    assert "app-pw" not in repr(cfg.email.gmail_app_password)
    assert "secret" not in repr(cfg.web_auth.password)
    # Round-trip via .get_secret_value() still returns the original value for real use.
    assert cfg.camera_secrets.password.get_secret_value() == "shared-pass"
    assert cfg.email.gmail_app_password.get_secret_value() == "app-pw"
    assert cfg.web_auth.password.get_secret_value() == "secret"


def test_env_file_provides_secrets_when_no_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pydantic-settings`` reads ``.env`` in cwd when env vars are unset.

    Other tests use ``monkeypatch.setenv`` which short-circuits the dotenv source — this is the one
    test that actually exercises ``env_file=".env"``.
    """
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path)
    # Clear host env so the .env file is the sole source — must not call ``_set_env`` here.
    for key in (
        "CAT_WATCHER_CAMERA_USERNAME",
        "CAT_WATCHER_CAMERA_PASSWORD",
        "CAT_WATCHER_GMAIL_USER",
        "CAT_WATCHER_GMAIL_APP_PASSWORD",
        "CAT_WATCHER_ALERT_TO_ADDRESSES",
        "CAT_WATCHER_WEB_USERNAME",
        "CAT_WATCHER_WEB_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    env_file_body = textwrap.dedent("""\
        CAT_WATCHER_CAMERA_USERNAME=env-file-user
        CAT_WATCHER_CAMERA_PASSWORD=env-file-pass
        CAT_WATCHER_GMAIL_USER=env-file-gmail@example.com
        CAT_WATCHER_GMAIL_APP_PASSWORD=env-file-app-pw
        CAT_WATCHER_ALERT_TO_ADDRESSES=env-file-recipient@example.com
        CAT_WATCHER_WEB_USERNAME=env-file-admin
        CAT_WATCHER_WEB_PASSWORD=env-file-secret
    """)
    _ = (tmp_path / ".env").write_text(env_file_body)

    cfg = load_config()

    assert cfg.camera_secrets.username == "env-file-user"
    assert cfg.camera_secrets.password.get_secret_value() == "env-file-pass"
    assert cfg.email.gmail_user == "env-file-gmail@example.com"
    assert cfg.email.gmail_app_password.get_secret_value() == "env-file-app-pw"
    assert cfg.email.alert_to_addresses == ("env-file-recipient@example.com",)
    assert cfg.web_auth.username == "env-file-admin"
    assert cfg.web_auth.password.get_secret_value() == "env-file-secret"


@pytest.mark.parametrize(
    "section",
    [
        "detector",
        "alerts",
        "alerts.email",
        "alerts.macos",
        "web",
        "storage",
        "retention",
        "backup",
        "poller",
    ],
)
def test_unknown_nested_field_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str) -> None:
    """``extra='forbid'`` applies to every sub-model, not just ``Config`` root."""
    body = _toml_with_field(section, "bogus_field", 42)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError, match=r"bogus_field"):
        _ = load_config()


def test_multiple_invalid_values_reported_in_one_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config with multiple errors surfaces all of them in the ``ConfigError`` message."""
    # Corrupt two unrelated fields so both errors are independent (no short-circuit):
    # detector.confidence_threshold above le=1.0, and backup.cadence_hour above le=23.
    body = _toml_with_field("detector", "confidence_threshold", 1.5)
    body = body + "\n[backup]\ncadence_hour = 24\n"
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError) as exc_info:
        _ = load_config()
    msg = str(exc_info.value)
    assert "confidence_threshold" in msg
    assert "cadence_hour" in msg


def test_duplicate_camera_names_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two ``[[cameras]]`` entries with the same name are rejected (storage slug collision)."""
    body = _VALID_TOML.replace('name = "office"', 'name = "pantry"')
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError, match=r"duplicate camera name"):
        _ = load_config()


def test_poller_cursor_guards_default_when_keys_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``[poller]`` omitting ``overlap_minutes`` / ``safety_net_hours`` falls back to defaults."""
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(_VALID_TOML)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    cfg = load_config()

    assert cfg.poller.overlap_minutes == 15
    assert cfg.poller.safety_net_hours == 6


def test_poller_cursor_guards_custom_values_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom ``overlap_minutes`` + ``safety_net_hours`` round-trip through the loader."""
    body = _VALID_TOML + "\n[poller]\noverlap_minutes = 5\nsafety_net_hours = 12\n"
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    cfg = load_config()

    assert cfg.poller.overlap_minutes == 5
    assert cfg.poller.safety_net_hours == 12


def test_poller_overlap_minutes_above_soft_cap_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``overlap_minutes`` well above the soft cap raises ``ConfigError`` mentioning the cap."""
    body = _toml_with_field("poller", "overlap_minutes", 9999)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    with pytest.raises(ConfigError, match=r"soft cap"):
        _ = load_config()


@pytest.mark.parametrize(
    ("overlap_minutes", "should_raise"),
    [
        # cadence_seconds=600 -> cap = (600 * 12) // 60 = 120.
        (120, False),
        (121, True),
    ],
)
def test_poller_overlap_soft_cap_scales_with_cadence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overlap_minutes: int,
    *,
    should_raise: bool,
) -> None:
    """The soft cap on ``overlap_minutes`` scales linearly with ``cadence_seconds``."""
    body = _VALID_TOML + f"\n[poller]\ncadence_seconds = 600\noverlap_minutes = {overlap_minutes}\n"
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    if should_raise:
        with pytest.raises(ConfigError):
            _ = load_config()
    else:
        cfg = load_config()
        assert cfg.poller.overlap_minutes == overlap_minutes


def test_empty_cameras_list_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A config with no ``[[cameras]]`` entries is rejected by ``Field(min_length=1)``."""
    body = textwrap.dedent("""\
        internal_root = "./data"
        storage_root = "./data"
        log_level = "INFO"

        [detector]
        model = "yolo11n.pt"
        confidence_threshold = 0.35
        frames_to_sample = 5

        [alerts]
        inactivity_hours = 12
        frequency_window_hours = 6
        frequency_threshold_count = 8
        cooldown_hours = 6

        [alerts.email]
        enabled = true
        smtp_host = "smtp.gmail.com"
        smtp_port = 587

        [alerts.macos]
        enabled = true

        [web]
        host = "0.0.0.0"
        port = 8000
        heartbeat_interval_seconds = 60
        public_url = "http://mac-mini.local:8000"
        display_timezone = "America/New_York"
    """)
    config_path = tmp_path / "config.toml"
    _ = config_path.write_text(body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(config_path))
    _set_env(monkeypatch)

    # Loose match: pydantic's exact "List should have at least 1 item" wording may shift versions.
    with pytest.raises(ConfigError, match=r"cameras"):
        _ = load_config()
