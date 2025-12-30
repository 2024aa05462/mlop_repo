import requests
import json
import logging
import os

logger = logging.getLogger(__name__)

class SlackNotifier:
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        
    def send_alert(self, title: str, message: str, severity: str = "info"):
        """
        Send a formatted alert to Slack
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not set. Skipping alert.")
            return

        color_map = {
            "info": "#36a64f",
            "warning": "#ffcc00",
            "error": "#ff0000",
            "critical": "#7a0000"
        }
        
        payload = {
            "attachments": [
                {
                    "color": color_map.get(severity, "#cccccc"),
                    "title": title,
                    "text": message,
                    "footer": "MLOps Monitoring System"
                }
            ]
        }
        
        try:
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

notifier = SlackNotifier()
