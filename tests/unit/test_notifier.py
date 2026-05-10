"""Tests for cat_watcher.notifier."""

import smtplib
import subprocess
from unittest.mock import MagicMock, create_autospec

import pytest
from pydantic import SecretStr

from cat_watcher.config import EmailRulesConfig, EmailSecrets, MacOsRulesConfig
from cat_watcher.notifier import (
    EmailResult,
    NotifResult,
    send_email,
    send_macos_notification,
)


def _email_secrets() -> EmailSecrets:
    """Build an ``EmailSecrets`` directly so init kwargs win over ambient env / .env."""
    return EmailSecrets(
        gmail_user="alerts@example.com",
        gmail_app_password=SecretStr("app-pw"),
        alert_to_addresses=("me@example.com", "other@example.com"),
    )


def _email_rules(*, enabled: bool = True) -> EmailRulesConfig:
    return EmailRulesConfig(enabled=enabled, smtp_host="smtp.gmail.com", smtp_port=587)


def _macos_rules(*, enabled: bool = True) -> MacOsRulesConfig:
    return MacOsRulesConfig(enabled=enabled)


def _patch_smtp(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Patch ``smtplib.SMTP`` and return ``(ctor, instance)``.

    The instance mock returns ``self`` from ``__enter__`` so ``with smtplib.SMTP(...) as smtp:``
    rebinds to the same object the assertions inspect.
    """
    smtp_instance = MagicMock(spec=smtplib.SMTP)
    smtp_instance.__enter__.return_value = smtp_instance
    smtp_instance.__exit__.return_value = False
    ctor: MagicMock = create_autospec(smtplib.SMTP, return_value=smtp_instance)
    monkeypatch.setattr(smtplib, "SMTP", ctor)
    return ctor, smtp_instance


def test_send_email_uses_smtp_starttls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path constructs the SMTP client, runs STARTTLS, logs in, and sends the message — pinning the full sequence."""
    ctor, fake_smtp = _patch_smtp(monkeypatch)

    result = send_email(
        subject="hello",
        body="world",
        secrets=_email_secrets(),
        rules=_email_rules(),
    )

    ctor.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
    fake_smtp.starttls.assert_called_once()
    fake_smtp.login.assert_called_once_with("alerts@example.com", "app-pw")
    fake_smtp.sendmail.assert_called_once()
    envelope_from, recipients, payload = fake_smtp.sendmail.call_args.args
    assert envelope_from == "alerts@example.com"
    assert sorted(recipients) == ["me@example.com", "other@example.com"]
    assert "Subject: hello" in payload
    assert "world" in payload
    # Mail providers (Gmail in particular) quarantine messages whose ``From:`` header doesn't match
    # the envelope sender; if From/To were swapped in the impl, no other test would catch it.
    assert "From: alerts@example.com" in payload
    assert "To: me@example.com, other@example.com" in payload
    assert isinstance(result, EmailResult)
    assert result.ok is True
    assert result.error is None


def test_send_email_uses_configured_smtp_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """SMTP host + port come from rules, not hardcoded defaults."""
    ctor, _ = _patch_smtp(monkeypatch)
    rules = EmailRulesConfig(enabled=True, smtp_host="mail.example.org", smtp_port=2525)

    _ = send_email(subject="s", body="b", secrets=_email_secrets(), rules=rules)

    ctor.assert_called_once_with("mail.example.org", 2525, timeout=30)


def test_send_email_returns_failure_on_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network-level failure (DNS, connection refused) returns ok=False rather than raising."""
    ctor: MagicMock = create_autospec(smtplib.SMTP, side_effect=OSError("name resolution failed"))
    monkeypatch.setattr(smtplib, "SMTP", ctor)

    result = send_email(subject="x", body="y", secrets=_email_secrets(), rules=_email_rules())

    assert result.ok is False
    assert result.error is not None
    assert "name resolution failed" in result.error


def test_send_email_returns_failure_on_smtp_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """``smtplib`` raising mid-conversation returns ok=False rather than propagating."""
    _, fake_smtp = _patch_smtp(monkeypatch)
    fake_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad creds")

    result = send_email(subject="x", body="y", secrets=_email_secrets(), rules=_email_rules())

    assert result.ok is False
    assert result.error is not None


def test_send_email_disabled_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rules.enabled=False`` returns ok=True with error="disabled"; SMTP is never touched."""
    ctor: MagicMock = create_autospec(
        smtplib.SMTP,
        side_effect=AssertionError("SMTP must not be constructed when disabled"),
    )
    monkeypatch.setattr(smtplib, "SMTP", ctor)

    result = send_email(subject="x", body="y", secrets=_email_secrets(), rules=_email_rules(enabled=False))

    ctor.assert_not_called()
    assert result.ok is True
    assert result.error == "disabled"


def _patch_subprocess_run(monkeypatch: pytest.MonkeyPatch, return_value: subprocess.CompletedProcess[bytes]) -> MagicMock:
    run_mock: MagicMock = create_autospec(subprocess.run, return_value=return_value)
    monkeypatch.setattr(subprocess, "run", run_mock)
    return run_mock


def test_send_macos_notification_invokes_osascript(monkeypatch: pytest.MonkeyPatch) -> None:
    """The happy path shells out to ``osascript -e 'display notification ...'`` with the title and body quoted."""
    completed: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    run_mock = _patch_subprocess_run(monkeypatch, completed)

    result = send_macos_notification(title="cat", body="seen", rules=_macos_rules())

    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "osascript"
    assert cmd[1] == "-e"
    assert "display notification" in cmd[2]
    assert '"seen"' in cmd[2]
    assert '"cat"' in cmd[2]
    assert isinstance(result, NotifResult)
    assert result.ok is True
    assert result.error is None


def test_send_macos_notification_escapes_double_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedded double-quotes in title/body are backslash-escaped (no AppleScript injection).

    Verifies the safety property by counting unescaped double-quotes in the rendered script: only
    the 4 wrapper quotes (around body literal + around title literal) may be unescaped. Any
    additional unescaped ``"`` would mean an input quote slipped through, letting the attacker
    close the literal and append arbitrary AppleScript.
    """
    completed: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    run_mock = _patch_subprocess_run(monkeypatch, completed)

    _ = send_macos_notification(
        title='evil"; do_bad_thing()',
        body='body"; rm -rf /',
        rules=_macos_rules(),
    )

    script: str = run_mock.call_args.args[0][2]
    unescaped_double_quotes = sum(1 for idx, ch in enumerate(script) if ch == '"' and (idx == 0 or script[idx - 1] != "\\"))
    assert unescaped_double_quotes == 4, f"expected only the 4 wrapper quotes unescaped, got {unescaped_double_quotes}: {script!r}"
    assert script.count('\\"') == 2


def test_send_macos_notification_escapes_backslashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone ``\\`` in the body must be escaped to ``\\\\`` so AppleScript doesn't parse-error.

    Stack-trace bodies (the BACKUP_STALE alert includes the last 50 lines of ``web.stderr.log``)
    routinely contain ``\\`` from Windows paths, repr() of bytes, etc. An unescaped backslash
    followed by an unrecognized escape character causes AppleScript to fail silently — the
    notification disappears, the operator never sees it.
    """
    completed: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    run_mock = _patch_subprocess_run(monkeypatch, completed)

    _ = send_macos_notification(title="path", body=r"C:\Users\op\file", rules=_macos_rules())

    script: str = run_mock.call_args.args[0][2]
    # Each ``\`` from the input becomes ``\\`` in the script. Three backslashes in the input ->
    # six in the rendered script.
    assert script.count("\\\\") == 3, f"expected three escaped backslashes, got {script!r}"
    # Every backslash must start a valid escape pair (``\\\\`` or ``\\"``). Walk the script,
    # consuming both chars of each escape so the second char isn't re-validated as a new backslash.
    idx = 0
    while idx < len(script):
        if script[idx] == "\\":
            assert idx + 1 < len(script), f"trailing backslash in {script!r}"
            assert script[idx + 1] in {"\\", '"'}, f"invalid escape \\{script[idx + 1]} at position {idx} in {script!r}"
            idx += 2
        else:
            idx += 1


def test_send_macos_notification_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-zero exit returns ok=False with stderr surfaced as error rather than raising."""
    completed: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"boom")
    _ = _patch_subprocess_run(monkeypatch, completed)

    result = send_macos_notification(title="t", body="b", rules=_macos_rules())

    assert result.ok is False
    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.parametrize(
    "subprocess_exception",
    [
        # Missing executable (e.g. running on Linux in CI without osascript on PATH).
        FileNotFoundError(2, "No such file or directory", "osascript"),
        # osascript hung past the timeout.
        subprocess.TimeoutExpired(cmd=["osascript"], timeout=10),
    ],
    ids=["missing-executable", "timeout"],
)
def test_send_macos_notification_handles_subprocess_exceptions(
    subprocess_exception: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both subprocess exceptions yield ok=False rather than propagating to the caller."""
    run_mock: MagicMock = create_autospec(subprocess.run, side_effect=subprocess_exception)
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = send_macos_notification(title="t", body="b", rules=_macos_rules())

    assert result.ok is False
    assert result.error is not None


def test_send_macos_notification_disabled_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rules.enabled=False`` returns ok=True with error="disabled"; subprocess is never invoked."""
    run_mock: MagicMock = create_autospec(
        subprocess.run,
        side_effect=AssertionError("subprocess.run must not be called when disabled"),
    )
    monkeypatch.setattr(subprocess, "run", run_mock)

    result = send_macos_notification(title="t", body="b", rules=_macos_rules(enabled=False))

    run_mock.assert_not_called()
    assert result.ok is True
    assert result.error == "disabled"
