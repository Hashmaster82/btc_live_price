"""Конфигурация приложения и константы."""
import os
from typing import Dict, Any, Optional
from enum import Enum
import json
import logging

# Настройка логирования
logging.basicConfig(
    filename='crypto_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Константы
CONFIG_FILE = "config.json"
LOCALES_DIR = "locales"
MAX_HISTORY_POINTS = 120
MAX_GUI_UPDATES_PER_SEC = 2
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/"
BINANCE_REST_URL = "https://api.binance.com/api/v3/ticker/price?symbol="
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = "YOUR_NEWS_API_KEY"
NEWS_UPDATE_INTERVAL = 300

# Список популярных пар
POPULAR_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt",
    "btceur", "etheur", "soleur",
    "btcrub", "ethrub", "solrub"
]


class TradingPair(Enum):
    BTC_USDT = "btcusdt"
    ETH_USDT = "ethusdt"
    SOL_USDT = "solusdt"
    BTC_EUR = "btceur"
    ETH_EUR = "etheur"
    SOL_EUR = "soleur"
    BTC_RUB = "btcrub"
    ETH_RUB = "ethrub"
    SOL_RUB = "solrub"


class AlertType(Enum):
    PRICE_UP = "price_up"
    PRICE_DOWN = "price_down"
    ABSOLUTE_UP = "absolute_up"
    ABSOLUTE_DOWN = "absolute_down"


class ConfigManager:
    """Менеджер конфигурации приложения."""

    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self._config = self._load_default_config()
        self._load_saved_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию по умолчанию."""
        return {
            "language": "auto",
            "symbol": "btcusdt",
            "portfolio_asset": "BTC",
            "portfolio_amount": 0.0,
            "alert_threshold_up": 1.0,
            "alert_threshold_down": 1.0,
            "alert_cooldown_sec": 15,
            "start_minimized": False,
            "always_on_top": False,
            "show_notifications": True,
            "show_sound": True,
            "color_price": True
        }

    def _load_saved_config(self) -> None:
        """Загружает сохраненную конфигурацию из файла."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                # Обновляем только существующие ключи
                for key, value in self._config.items():
                    if key in loaded_config:
                        self._config[key] = loaded_config[key]
            except Exception as e:
                logging.error(f"Config load error: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение конфигурации."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Устанавливает значение конфигурации."""
        self._config[key] = value
        self._save_config()

    def update(self, new_config: Dict[str, Any]) -> None:
        """Обновляет несколько значений конфигурации."""
        self._config.update(new_config)
        self._save_config()

    def _save_config(self) -> None:
        """Сохраняет конфигурацию в файл."""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Config save error: {e}")

    @property
    def all_config(self) -> Dict[str, Any]:
        """Возвращает всю конфигурацию."""
        return self._config.copy()