# notifiers.py
import requests

class TelegramNotifier:
    def __init__(self, token: str, chat_id: int):
        self.token = token
        self.chat_id = int(chat_id)

    def send(self, text: str):
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {"chat_id": self.chat_id,
                       "text": text,
                       "disable_web_page_preview": True}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print("telegram notify failed:", e)
