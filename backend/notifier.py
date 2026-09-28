"""Telegram & Webhook Alert Notification Engine with Threaded Dispatch."""

import os
import time
import requests
import threading
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("SecuritySystem.Notifier")

class AlertNotifier:
    """Dispatches instant incident alerts via Telegram Bot API and generic Webhooks."""

    def __init__(
        self,
        telegram_enabled: bool = False,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        webhook_enabled: bool = False,
        webhook_url: str = "",
        global_cooldown: float = 3.0
    ):
        self.telegram_enabled = telegram_enabled
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.webhook_enabled = webhook_enabled
        self.webhook_url = webhook_url
        self.global_cooldown = global_cooldown
        self.last_notification_time = 0.0

    def update_credentials(
        self,
        telegram_enabled: Optional[bool] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        webhook_enabled: Optional[bool] = None,
        webhook_url: Optional[str] = None
    ):
        """Updates configuration at runtime."""
        if telegram_enabled is not None:
            self.telegram_enabled = telegram_enabled
        if telegram_token is not None:
            self.telegram_token = telegram_token
        if telegram_chat_id is not None:
            self.telegram_chat_id = telegram_chat_id
        if webhook_enabled is not None:
            self.webhook_enabled = webhook_enabled
        if webhook_url is not None:
            self.webhook_url = webhook_url

    def dispatch_alert(self, event_data: Dict[str, Any], snapshot_path: Optional[str] = None):
        """
        Dispatches alert in a background daemon thread to avoid blocking video inference.
        """
        now = time.time()
        if (now - self.last_notification_time) < self.global_cooldown:
            return  # Throttled

        self.last_notification_time = now

        # Run dispatch asynchronously
        t = threading.Thread(target=self._send_async, args=(event_data, snapshot_path), daemon=True)
        t.start()

    def _send_async(self, event_data: Dict[str, Any], snapshot_path: Optional[str] = None):
        """Internal worker executing HTTP calls."""
        msg = (
            f"🚨 *SECURITY ALERT: {event_data.get('event_type', 'VIOLATION')}* 🚨\n\n"
            f"📍 *Zone:* `{event_data.get('zone_name', 'N/A')}`\n"
            f"👤 *Target ID:* `#{event_data.get('track_id', 'N/A')}`\n"
            f"⏱ *Duration:* `{event_data.get('duration', 0.0)}s`\n"
            f"🎯 *Confidence:* `{int(event_data.get('confidence', 0.0) * 100)}%`\n"
            f"💬 *Details:* {event_data.get('message', '')}"
        )

        if self.telegram_enabled and self.telegram_token and self.telegram_chat_id:
            self._send_telegram(msg, snapshot_path)

        if self.webhook_enabled and self.webhook_url:
            self._send_webhook(event_data, snapshot_path)

    def _send_telegram(self, text: str, snapshot_path: Optional[str] = None) -> bool:
        """Sends Telegram message or photo via Telegram Bot API."""
        try:
            if snapshot_path and os.path.exists(snapshot_path):
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(snapshot_path, "rb") as photo:
                    resp = requests.post(
                        url,
                        data={"chat_id": self.telegram_chat_id, "caption": text, "parse_mode": "Markdown"},
                        files={"photo": photo},
                        timeout=8.0
                    )
            else:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                resp = requests.post(
                    url,
                    json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"},
                    timeout=8.0
                )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram alert delivery failed: {e}")
            return False

    def _send_webhook(self, event_data: Dict[str, Any], snapshot_path: Optional[str] = None) -> bool:
        """Sends HTTP POST webhook payload."""
        try:
            payload = {
                "event": "security_alert",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data": event_data,
                "snapshot_available": bool(snapshot_path and os.path.exists(snapshot_path))
            }
            resp = requests.post(self.webhook_url, json=payload, timeout=5.0)
            return resp.status_code in (200, 201, 204)
        except Exception as e:
            logger.error(f"Webhook alert delivery failed: {e}")
            return False

    def test_telegram(self, token: str, chat_id: str) -> Tuple[bool, str]:
        """Validates Telegram credentials by sending a test ping."""
        if not token or not chat_id:
            return False, "Token and Chat ID cannot be empty."
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": "🛡️ *Intelligent Security System Advanced*\n\n✅ Telegram alert channel verified successfully!",
                    "parse_mode": "Markdown"
                },
                timeout=6.0
            )
            data = resp.json()
            if data.get("ok"):
                return True, "Telegram test message sent successfully!"
            else:
                return False, f"Telegram API Error: {data.get('description', 'Unknown error')}"
        except Exception as e:
            return False, f"Network Error: {str(e)}"

    def test_webhook(self, webhook_url: str) -> Tuple[bool, str]:
        """Validates webhook connectivity."""
        if not webhook_url:
            return False, "Webhook URL cannot be empty."
        try:
            payload = {
                "event": "test_ping",
                "message": "Security system webhook test successful.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            resp = requests.post(webhook_url, json=payload, timeout=5.0)
            if resp.status_code in (200, 201, 204):
                return True, f"Webhook accepted payload (HTTP {resp.status_code})!"
            else:
                return False, f"Webhook returned HTTP {resp.status_code}"
        except Exception as e:
            return False, f"Webhook connection failed: {str(e)}"
