#!/usr/bin/env python3
"""Email a failed systemd unit's recent journal. Wired up via OnFailure=ai-stack-notify@%n.service.

Needs in ~/ai-stack/.env (Gmail: create an App Password at https://myaccount.google.com/apppasswords,
2-Step Verification must be on — a normal account password will not work):
    NOTIFY_EMAIL_TO=you@example.com
    SMTP_USER=sender@example.com
    SMTP_PASSWORD=<16-char app password>
    # optional: SMTP_HOST=smtp.gmail.com  SMTP_PORT=465
Without them it logs "not configured" and exits 0 (so a missing config never masks the real failure).
Test: scripts/notify-failure.py --test
"""
import os, smtplib, socket, subprocess, sys
from email.message import EmailMessage

ENV = os.path.expanduser("~/ai-stack/.env")


def load_env():
    cfg = {}
    try:
        for line in open(ENV):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return cfg


def send_email(subject: str, body: str) -> str:
    """Send one email using the .env settings. Returns a status line; never raises, so a mail
    problem can't break the caller (also used by scripts/model-watchdog.py)."""
    cfg = load_env()
    to, user, pw = cfg.get("NOTIFY_EMAIL_TO"), cfg.get("SMTP_USER"), cfg.get("SMTP_PASSWORD")
    if not (to and user and pw):
        return (f"notify: email not configured in {ENV} (NOTIFY_EMAIL_TO/SMTP_USER/SMTP_PASSWORD), "
                f"not sent: {subject}")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(cfg.get("SMTP_HOST", "smtp.gmail.com"), int(cfg.get("SMTP_PORT", "465")),
                              timeout=30) as s:
            s.login(user, pw)
            s.send_message(msg)
    except Exception as e:
        return f"notify: sending failed ({e}): {subject}"
    return f"notify: emailed {to}: {subject}"


def main():
    unit = sys.argv[1] if len(sys.argv) > 1 else "unknown.service"
    host = socket.gethostname()
    if unit == "--test":
        subject, body = f"[ai-stack] test notification from {host}", "Notification path works."
    else:
        log = subprocess.run(["journalctl", "-u", unit, "-n", "60", "--no-pager", "-o", "cat"],
                             capture_output=True, text=True).stdout
        subject = f"[ai-stack] {unit} FAILED on {host}"
        body = (f"{unit} failed on {host}.\n\nCheck: systemctl status {unit}; "
                f"journalctl -u {unit}\nRecovery docs: ~/ai-stack/docs/RECOVERY.md\n\n"
                f"--- last 60 journal lines ---\n{log}")
    print(send_email(subject, body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
