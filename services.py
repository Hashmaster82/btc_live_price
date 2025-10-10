"""Сервисы для работы с внешними API."""
import json
import time
import threading
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

import websocket
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import BINANCE_WS_URL, BINANCE_REST_URL, BINANCE_KLINES_URL
from models import PriceUpdate


class PriceProvider(ABC):
    """Абстрактный поставщик цен."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def start_stream(self, symbol: str, callback: Callable) -> bool:
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        pass


class BinancePriceProvider(PriceProvider):
    """Поставщик цен с Binance."""

    def __init__(self):
        self.ws = None
        self.callback = None
        self.running = False
        self.thread = None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Получает текущую цену через REST API."""
        try:
            url = f"{BINANCE_REST_URL}{symbol.upper()}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except Exception as e:
            logging.error(f"REST API error: {e}")
        return None

    def start_stream(self, symbol: str, callback: Callable) -> bool:
        """Запускает WebSocket поток."""
        if self.running:
            self.stop_stream()

        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._run_websocket, args=(symbol,), daemon=True)
        self.thread.start()
        return True

    def stop_stream(self) -> None:
        """Останавливает WebSocket поток."""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def _run_websocket(self, symbol: str) -> None:
        """Запускает WebSocket соединение."""
        retry_delay = 5

        while self.running:
            try:
                def on_message(ws, message):
                    if not self.running:
                        return
                    try:
                        data = json.loads(message)
                        price = float(data['c'])
                        if self.callback:
                            self.callback(price, time.time())
                    except Exception as e:
                        logging.error(f"WebSocket message error: {e}")

                def on_error(ws, error):
                    if self.running and self.callback:
                        self.callback(None, time.time(), error=str(error))

                def on_close(ws, code, msg):
                    if self.running and self.callback:
                        self.callback(None, time.time(), disconnected=True)

                def on_open(ws):
                    if self.running and self.callback:
                        self.callback(None, time.time(), connected=True)

                ws_url = f"{BINANCE_WS_URL}{symbol.lower()}@ticker"
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )

                self.ws.run_forever(ping_interval=30, ping_timeout=10)

            except Exception as e:
                logging.error(f"WebSocket error: {e}")

            if self.running:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 60)

        import feedparser
        from typing import List
        import html

class NewsService:
            """Служба получения новостей через RSS (без API-ключа)."""

            def __init__(self, api_key: str = None):
                # CoinDesk RSS — один из самых надёжных источников по крипте
                self.rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"

            def fetch_news(self, language: str = "en") -> List[str]:
                """
                Получает последние заголовки из RSS.
                Поддержка языка ограничена — CoinDesk в основном на английском.
                """
                if language != "en":
                    # Можно добавить другие источники для ru, но пока оставим так
                    return ["Новости на русском пока недоступны"]

                try:
                    feed = feedparser.parse(self.rss_url)
                    if feed.bozo and not feed.entries:
                        raise ValueError("Некорректный RSS-канал")

                    headlines = []
                    for entry in feed.entries[:3]:  # Берём 3 свежие новости
                        # Очистка от HTML-сущностей (например, &amp; → &)
                        title = html.unescape(entry.title)
                        headlines.append(title.strip())
                    return headlines
                except Exception as e:
                    # Логирование ошибки (в реальном приложении — через ErrorHandler)
                    print(f"[NewsService] Ошибка загрузки RSS: {e}")
                    return ["Не удалось загрузить новости"]

class NotificationService:
    """Сервис уведомлений."""

    def __init__(self):
        self._init_platform_specific()

    def _init_platform_specific(self) -> None:
        """Инициализирует платформо-специфичные функции."""
        try:
            import winsound
            self._play_sound = winsound.MessageBeep
            self.has_sound = True
        except ImportError:
            self._play_sound = lambda: print('\a', end='', flush=True)
            self.has_sound = True

        try:
            from plyer import notification
            self._show_notification = notification.notify
            self.has_notifications = True
        except ImportError:
            self._show_notification = None
            self.has_notifications = False

    def play_alert(self) -> None:
        """Воспроизводит звук оповещения."""
        if self.has_sound:
            try:
                self._play_sound()
            except Exception as e:
                logging.error(f"Sound error: {e}")

    def show_notification(self, title: str, message: str, timeout: int = 5) -> None:
        """Показывает системное уведомление."""
        if self.has_notifications and self._show_notification:
            try:
                self._show_notification(
                    title=title,
                    message=message,
                    app_name="Crypto Tracker",
                    timeout=timeout
                )
            except Exception as e:
                logging.error(f"Notification error: {e}")