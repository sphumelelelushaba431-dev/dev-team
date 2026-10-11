"""LINE Official Account webhook (optional messaging channel)."""
import base64
import hashlib
import hmac

import requests

from .. import config


def valid_signature(body: bytes, signature: str) -> bool:
    if not config.LINE_CHANNEL_SECRET:
        return False
    digest = hmac.new(config.LINE_CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature or "")


def reply(reply_token: str, text: str):
    requests.post("https://api.line.me/v2/bot/message/reply",
                  headers={"Authorization": f"Bearer {config.LINE_CHANNEL_ACCESS_TOKEN}"},
                  json={"replyToken": reply_token, "messages": [{"type": "text", "text": text[:4900]}]},
                  timeout=10)
