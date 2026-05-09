# Outbound email setup

cat-watcher delivers operator alerts (inactivity, unusual frequency, agent
failures, storage problems) over two channels: email and macOS notifications.
This doc covers the email side.

The sender targets **Gmail SMTP over STARTTLS**. The host and port are
configurable, but the protocol is not — any provider you point it at must accept
`STARTTLS` on the configured port. SSL-on-connect endpoints (typically port 465)
will not work.

## 1. Prerequisites

- A Gmail account you control.
- **2-Step Verification enabled** on that account. Google only exposes app
  passwords to accounts with 2SV on.
- An app password minted at <https://myaccount.google.com/apppasswords>. Google
  shows the 16-character password **once**; copy it and strip the spaces Google
  inserts for readability before you paste it into `.env`.

## 2. Fill `.env`

Three env vars in `.env` (copy from `.env.example` first if you haven't):

```env
CAT_WATCHER_GMAIL_USER=you@gmail.com
CAT_WATCHER_GMAIL_APP_PASSWORD=abcdefghijklmnop
CAT_WATCHER_ALERT_TO_ADDRESSES=you@gmail.com,partner@example.com
```

- `CAT_WATCHER_GMAIL_USER` — the Gmail address. Used both as the SMTP auth
  username and as the `From:` header on every alert.
- `CAT_WATCHER_GMAIL_APP_PASSWORD` — the 16-character app password from §1, no
  spaces.
- `CAT_WATCHER_ALERT_TO_ADDRESSES` — comma-separated recipient list. At least
  one address is required; sending to yourself is fine.

## 3. (Optional) override SMTP host or port

The defaults are `smtp.gmail.com:587`. To point at a different host (for
example, a Workspace relay), add an `[alerts.email]` block to `config.toml`:

```toml
[alerts.email]
smtp_host = "smtp-relay.gmail.com"
smtp_port = 587
```

Both keys are independently optional — omit either to keep its default. The
chosen port must speak `STARTTLS`; the sender does not support SSL-on-connect.

## 4. Verify

Send one synthetic alert through both channels:

```bash
pixi run cat-watcher test-notification
```

The command prints a per-channel result. If the email leg reports failure, the
three most common causes are:

1. The app password was copied with spaces, or with stray whitespace.
2. 2-Step Verification isn't actually enabled on the account, so the password
   minted in §1 is rejected.
3. Outbound port 587 is blocked by the local network (some ISPs and coffee-shop
   Wi-Fi do this).

The macOS notification leg has no dependency on §1–3; it is a useful cross-check
that the alerts agent itself is wired up.

## 5. Disable email alerts

To stop sending email without removing the credentials, set:

```toml
[alerts.email]
enabled = false
```

The alerts agent still evaluates rules and still writes `alerts_sent` rows, and
macOS notifications still fire. Only the SMTP send is skipped — the sender
short-circuits with a success result so the agent records the alert the same way
it would on a real send.

## Reference

| Setting                          | Where         | Default          | Meaning                                                |
| -------------------------------- | ------------- | ---------------- | ------------------------------------------------------ |
| `CAT_WATCHER_GMAIL_USER`         | `.env`        | —                | Gmail address; SMTP auth username and `From:` header   |
| `CAT_WATCHER_GMAIL_APP_PASSWORD` | `.env`        | —                | 16-char Google app password (no spaces)                |
| `CAT_WATCHER_ALERT_TO_ADDRESSES` | `.env`        | —                | Comma-separated recipient list (≥ 1 address)           |
| `[alerts.email].enabled`         | `config.toml` | `true`           | When `false`, `send_email` short-circuits with success |
| `[alerts.email].smtp_host`       | `config.toml` | `smtp.gmail.com` | SMTP host; must accept `STARTTLS` on `smtp_port`       |
| `[alerts.email].smtp_port`       | `config.toml` | `587`            | SMTP port; SSL-on-connect (e.g. 465) is not supported  |
