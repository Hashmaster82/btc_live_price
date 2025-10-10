"""Вспомогательные утилиты."""
import os
import json
import locale
import math
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging


class FormatUtils:
    """Утилиты форматирования."""

    @staticmethod
    def format_price(price: float, currency: str = "USD") -> str:
        """Форматирует цену для отображения."""
        if currency == "RUB":
            return f"₽{price:,.0f}"
        elif currency == "EUR":
            return f"€{price:,.2f}"
        else:
            return f"${price:,.2f}"

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Форматирует timestamp в читаемую дату."""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    @staticmethod
    def format_portfolio_value(amount: float, asset: str, btc_price: float, usdt_rub: float) -> str:
        """Форматирует значение портфеля."""
        if amount <= 0:
            return ""

        if asset == "BTC":
            rub_value = amount * btc_price * usdt_rub
        else:
            rub_value = amount * usdt_rub

        return f"Оборот: ₽{rub_value:,.0f}"

class Internationalization:
    """Класс для интернационализации."""

    def __init__(self, locale_dir: str = "locales"):
        self.locale_dir = locale_dir
        self.current_language = "en"
        self.translations: Dict[str, str] = {}
        self._load_system_language()

    def _load_system_language(self) -> None:
        """Определяет язык системы."""
        sys_lang = locale.getdefaultlocale()[0]
        self.current_language = "ru" if sys_lang and sys_lang.startswith("ru") else "en"

    def set_language(self, language: str) -> None:
        """Устанавливает язык."""
        if language == "auto":
            self._load_system_language()
        else:
            self.current_language = language
        self._load_translations()

    def _load_translations(self) -> None:
        """Загружает переводы для текущего языка."""
        path = os.path.join(self.locale_dir, f"{self.current_language}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.translations = json.load(f)
                return
            except Exception as e:
                logging.error(f"Translation load error: {e}")

        # Загрузка переводов по умолчанию
        self.translations = self._get_default_translations()

    def _get_default_translations(self) -> Dict[str, str]:
        """Возвращает переводы по умолчанию."""
        return {
            "title": "Crypto Price Tracker",
            "connecting": "Connecting...",
            "connected": "Connected",
            "disconnected": "Disconnected",
            "error": "Connection error",
            "live_updated": "Live • Updated",
            "settings_title": "Settings",
            "language": "Language",
            "trading_pair": "Trading Pair",
            "alert_threshold_up": "Alert if price ↑ (%)",
            "alert_threshold_down": "Alert if price ↓ (%)",
            "alert_absolute_up": "Alert if price ≥",
            "alert_absolute_down": "Alert if price ≤",
            "sound_enabled": "Sound Alert",
            "notifications_enabled": "System Notifications",
            "display_options": "Display Options",
            "color_price": "Color price based on change",
            "save": "Save",
            "btn_minimize": "Minimize to Tray",
            "btn_always_on_top": "Always on Top",
            "btn_settings": "Settings",
            "tray_show": "Show",
            "tray_exit": "Exit",
            "alert_message": "Price alert: {msg}",
            "history_title": "Alert History",
            "converter_title": "Converter",
            "enter_amount": "Enter amount:",
            "chart_title": "Price Chart"
        }

    def get_plural(self, key: str, count: int, **kwargs) -> str:
        """Получает переведенный текст с учетом множественного числа."""
        template = self.translations.get(key, key)

        # Простая логика для русского языка
        if self.current_language == "ru":
            if count % 10 == 1 and count % 100 != 11:
                template = template.replace("{count}", f"{count}")
            elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
                template = template.replace("{count}", f"{count}")
            else:
                template = template.replace("{count}", f"{count}")

        try:
            return template.format(**kwargs)
        except Exception:
            return template

    def get_text(self, key: str, **kwargs) -> str:
        """Получает переведенный текст."""
        template = self.translations.get(key, key)
        try:
            return template.format(**kwargs)
        except Exception:
            return template


class ErrorHandler:
    """Обработчик ошибок."""

    @staticmethod
    def handle_websocket_error(error: Exception, context: str = "") -> None:
        """Обрабатывает ошибки WebSocket."""
        logging.error(f"WebSocket error in {context}: {error}")

    @staticmethod
    def handle_api_error(response, context: str = "") -> Optional[Dict]:
        """Обрабатывает ошибки API."""
        if response.status_code != 200:
            logging.error(f"API error in {context}: {response.status_code}")
            return None
        try:
            return response.json()
        except Exception as e:
            logging.error(f"API JSON parse error in {context}: {e}")
            return None

    @staticmethod
    def handle_ui_error(error: Exception, context: str = "") -> None:
        """Обрабатывает ошибки UI."""
        logging.error(f"UI error in {context}: {error}")


class FormatUtils:
    """Утилиты форматирования."""

    @staticmethod
    def format_price(price: float, currency: str = "USD") -> str:
        """Форматирует цену для отображения."""
        if currency == "RUB":
            return f"₽{price:,.0f}"
        elif currency == "EUR":
            return f"€{price:,.2f}"
        else:
            return f"${price:,.2f}"

    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Форматирует timestamp в читаемую дату."""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

    @staticmethod
    def format_portfolio_value(amount: float, asset: str, btc_price: float, usdt_rub: float) -> str:
        """Форматирует значение портфеля."""
        if amount <= 0:
            return ""

        if asset == "BTC":
            rub_value = amount * btc_price * usdt_rub
        else:
            rub_value = amount * usdt_rub

        return f"Оборот: ₽{rub_value:,.0f}"


class ValidationUtils:
    """Утилиты валидации."""

    @staticmethod
    def validate_price(price: Any) -> bool:
        """Валидирует цену."""
        try:
            price_float = float(price)
            return price_float > 0 and not math.isnan(price_float) and math.isfinite(price_float)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_threshold(threshold: Any) -> bool:
        """Валидирует пороговое значение."""
        try:
            threshold_float = float(threshold)
            return 0 < threshold_float <= 100
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_amount(amount: Any) -> bool:
        """Валидирует количество."""
        try:
            amount_float = float(amount)
            return amount_float >= 0
        except (ValueError, TypeError):
            return False