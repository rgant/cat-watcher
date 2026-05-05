"""Outbound email + macOS notification senders.

These are pure I/O wrappers — no rule evaluation, no DB writes, no cool-down logic. The alerts
agent (:mod:`cat_watcher.alerts`) decides *when* to send; this module decides *how*. Each function
returns a typed result and never raises through to the caller; failures are recorded on the
``alerts_sent`` row by the alerts agent.

A "disabled" channel (``rules.enabled is False``) is a valid final state, not a failure. Both
senders short-circuit before touching SMTP / ``osascript`` and return ``ok=True`` with
``error="disabled"`` so the alerts agent can record the explicit no-op without rendering the
delivery failed.
"""

import logging
import smtplib
import subprocess
from dataclasses import dataclass
from email.message import EmailMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cat_watcher.config import EmailRulesConfig, EmailSecrets, MacOsRulesConfig


logger = logging.getLogger(__name__)

_SMTP_TIMEOUT_SECONDS = 30
_OSASCRIPT_TIMEOUT_SECONDS = 10
_DISABLED_MARKER = "disabled"


@dataclass(frozen=True)
class EmailResult:
    """Outcome of a :func:`send_email` call.

    ``ok=True, error=None`` -> delivered. ``ok=True, error="disabled"`` -> channel disabled, no
    delivery attempted (intentional no-op). ``ok=False`` -> delivery failed; ``error`` is the short
    reason (used in the ``alerts_sent.delivery_error`` column).
    """

    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class NotifResult:
    """Outcome of a :func:`send_macos_notification` call. Same shape as :class:`EmailResult`."""

    ok: bool
    error: str | None = None


def send_email(
    subject: str,
    body: str,
    *,
    secrets: EmailSecrets,
    rules: EmailRulesConfig,
) -> EmailResult:
    """Send a plain-text email via SMTP+STARTTLS.

    Returns :class:`EmailResult` rather than raising so the alerts agent can record success and
    failure uniformly. Both ``OSError`` (network / DNS / TLS) and ``smtplib.SMTPException`` (auth,
    refused, malformed) are caught.
    """
    if not rules.enabled:
        return EmailResult(ok=True, error=_DISABLED_MARKER)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = secrets.gmail_user
    msg["To"] = ", ".join(secrets.alert_to_addresses)
    msg.set_content(body)

    try:
        with smtplib.SMTP(rules.smtp_host, rules.smtp_port, timeout=_SMTP_TIMEOUT_SECONDS) as smtp:
            _ = smtp.starttls()
            _ = smtp.login(secrets.gmail_user, secrets.gmail_app_password.get_secret_value())
            _ = smtp.sendmail(
                secrets.gmail_user,
                list(secrets.alert_to_addresses),
                msg.as_string(),
            )
    except (OSError, smtplib.SMTPException) as exc:
        logger.exception("email send failed (host=%s port=%d)", rules.smtp_host, rules.smtp_port)
        return EmailResult(ok=False, error=str(exc) or exc.__class__.__name__)
    return EmailResult(ok=True)


def _escape_for_applescript(value: str) -> str:
    """Escape a Python string for embedding in an AppleScript double-quoted literal.

    AppleScript string literals recognize ``\\\\``, ``\\"``, ``\\n``, ``\\r``, ``\\t``. An unescaped
    backslash followed by an unrecognized character is undefined (parse error on some macOS
    versions, silent mangling on others) — both manifest as the notification disappearing without
    operator visibility. Order matters: escape backslashes *first* so we don't double-escape the
    backslash we add for the quote.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _osascript_payload(title: str, body: str) -> str:
    """Render the AppleScript ``display notification`` command with escaped operator content."""
    return f'display notification "{_escape_for_applescript(body)}" with title "{_escape_for_applescript(title)}"'


def send_macos_notification(
    title: str,
    body: str,
    *,
    rules: MacOsRulesConfig,
) -> NotifResult:
    """Trigger a macOS user-notification banner via ``osascript``.

    Returns :class:`NotifResult` and never raises. Handles missing executable (``osascript`` absent
    on non-macOS hosts), timeout, and non-zero exit. The first call after install triggers the
    system "send notifications" permission prompt — that's why the umbrella CLI's
    ``test-notification`` sub-command (Task 25) exists.
    """
    if not rules.enabled:
        return NotifResult(ok=True, error=_DISABLED_MARKER)

    cmd = ["osascript", "-e", _osascript_payload(title, body)]
    try:
        result = subprocess.run(  # noqa: S603  # cmd is fully constructed, not user-shell-evaluated
            cmd,
            check=False,
            capture_output=True,
            timeout=_OSASCRIPT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.exception("osascript invocation failed")
        return NotifResult(ok=False, error=str(exc) or exc.__class__.__name__)

    if result.returncode != 0:
        err = result.stderr.decode(errors="replace").strip() or f"osascript exit {result.returncode}"
        logger.error("osascript returned %d: %s", result.returncode, err)
        return NotifResult(ok=False, error=err)
    return NotifResult(ok=True)
