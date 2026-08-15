# Outbound email setup

cat-watcher delivers operator alerts (inactivity, unusual frequency, agent
failures, storage problems) over two channels: email and macOS notifications.
This doc covers the email side.

The sender targets **Gmail SMTP over STARTTLS**. You can configure the host and
the port. You cannot change the protocol. The provider must accept `STARTTLS` on
the configured port. An SSL-on-connect endpoint (port 465 usually) does not
work.

## 1. Prerequisites

- A Gmail account you control.
- **2-Step Verification enabled** on that account. Google exposes app passwords
  only to an account with 2SV on.
- An app password created at <https://myaccount.google.com/apppasswords>. Google
  shows the 16-character password one time. Google inserts spaces for
  readability, so strip them before you paste the password into `.env`.

## 2. Fill `.env`

Set these variables in `.env`. If the file does not exist, copy `.env.example`
first.

```env
CAT_WATCHER_GMAIL_USER=you@gmail.com
CAT_WATCHER_GMAIL_APP_PASSWORD=abcdefghijklmnop
CAT_WATCHER_ALERT_TO_ADDRESSES=you@gmail.com,partner@example.com
```

- `CAT_WATCHER_GMAIL_USER`: the Gmail address. It is the SMTP auth username. It
  is also the `From:` header on every alert.
- `CAT_WATCHER_GMAIL_APP_PASSWORD`: the 16-character app password from §1, no
  spaces.
- `CAT_WATCHER_ALERT_TO_ADDRESSES`: comma-separated recipient list. One address
  is enough. Your own address works.

## 3. (Optional) override SMTP host or port

The defaults are `smtp.gmail.com:587`. To point at a different host (for
example, a Workspace relay), add an `[alerts.email]` block to `config.toml`:

```toml
[alerts.email]
smtp_host = "smtp-relay.gmail.com"
smtp_port = 587
```

Each key is optional. If you omit a key, it keeps its default. The port must
accept `STARTTLS`. The sender does not support SSL-on-connect.

## 4. Verify

Send one synthetic alert through the email channel and the macOS channel:

```bash
pixi run cat-watcher test-notification
```

The command prints a per-channel result. If the email leg reports failure, look
at these causes:

1. The app password was copied with spaces, or with stray whitespace.
2. 2-Step Verification is not enabled on the account, so Google rejects the
   password from §1.
3. The local network blocks outbound port 587. Some ISPs and coffee-shop Wi-Fi
   do this.

The macOS notification leg does not depend on §1–3. It cross-checks that the
alerts agent itself works.

## 5. Disable email alerts

To stop sending email without removing the credentials, set:

```toml
[alerts.email]
enabled = false
```

The alerts agent still evaluates rules and still writes `alerts_sent` rows, and
macOS notifications still fire. Only the SMTP send is skipped. The sender
returns a success result at once, so the agent records the alert exactly as it
records a real send.

## Reference

| Setting                          | Where         | Default          | Meaning                                                |
| -------------------------------- | ------------- | ---------------- | ------------------------------------------------------ |
| `CAT_WATCHER_GMAIL_USER`         | `.env`        | none             | Gmail address. SMTP auth username and `From:` header   |
| `CAT_WATCHER_GMAIL_APP_PASSWORD` | `.env`        | none             | 16-char Google app password (no spaces)                |
| `CAT_WATCHER_ALERT_TO_ADDRESSES` | `.env`        | none             | Comma-separated recipient list (≥ 1 address)           |
| `[alerts.email].enabled`         | `config.toml` | `true`           | When `false`, `send_email` short-circuits with success |
| `[alerts.email].smtp_host`       | `config.toml` | `smtp.gmail.com` | SMTP host. Must accept `STARTTLS` on `smtp_port`       |
| `[alerts.email].smtp_port`       | `config.toml` | `587`            | SMTP port. SSL-on-connect (port 465) is not supported  |
