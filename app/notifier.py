"""Escalation notifications (Twilio SMS/WhatsApp when configured, otherwise a log line)."""
import logging

import requests

from . import config

log = logging.getLogger(__name__)


def notify_escalation(complaint: dict):
    msg = (f"CampusVoice escalation {complaint['reference']}: {complaint['department']} "
           f"[{complaint['urgency']}] - {complaint['status_reason']}")
    if not (config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN and config.TWILIO_FROM and config.TWILIO_NOTIFY_TO):
        log.info("ESCALATION (Twilio not configured): %s", msg)
        return False
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{config.TWILIO_ACCOUNT_SID}/Messages.json",
            data={"From": config.TWILIO_FROM, "To": config.TWILIO_NOTIFY_TO, "Body": msg},
            auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN), timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        log.warning("Twilio notification failed: %s", exc)
        return False
