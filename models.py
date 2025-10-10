"""Модели данных и бизнес-логика."""
import time
import math
from typing import Deque, Dict, List, Optional, Tuple, Callable
from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from config import MAX_HISTORY_POINTS


@dataclass
class PriceUpdate:
    """Модель обновления цены."""
    timestamp: float
    price: float
    volume: float = 0.0
    symbol: str = ""


@dataclass
class AlertConfig:
    """Конфигурация оповещений."""
    threshold_up: float
    threshold_down: float
    absolute_up: Optional[float] = None
    absolute_down: Optional[float] = None
    cooldown_sec: int = 15


class Observable:
    """Базовый класс для реализации паттерна Observer."""

    def __init__(self):
        self._observers: List[Callable] = []

    def add_observer(self, observer: Callable) -> None:
        """Добавляет наблюдателя."""
        self._observers.append(observer)

    def remove_observer(self, observer: Callable) -> None:
        """Удаляет наблюдателя."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, event: str, data: Dict) -> None:
        """Уведомляет всех наблюдателей."""
        for observer in self._observers:
            try:
                observer(event, data)
            except Exception as e:
                print(f"Observer error: {e}")


class PriceMonitor(Observable):
    """Мониторинг цен с системой оповещений."""

    def __init__(self, symbol: str = "btcusdt"):
        super().__init__()
        self.symbol = symbol.lower()
        self.running = True
        self.price_history: Deque[Tuple[float, float]] = deque(maxlen=MAX_HISTORY_POINTS)
        self.last_alert_time = 0
        self.last_price: Optional[float] = None

    def add_price(self, price: float, volume: float = 0) -> None:
        """Добавляет новую цену в историю."""
        if not self.running or not self._is_valid_price(price):
            return

        now = time.time()
        self.price_history.append((now, price))
        self.last_price = price

        # Уведомляем наблюдателей
        self.notify_observers('price_update', {
            'price': price,
            'timestamp': now,
            'symbol': self.symbol
        })

    def _is_valid_price(self, price: float) -> bool:
        """Проверяет валидность цены."""
        return price > 0 and math.isfinite(price)

    def check_alert(self, current_price: float, current_time: float, config: AlertConfig) -> Tuple[bool, str]:
        """Проверяет условия для оповещения."""
        if not self.price_history or len(self.price_history) < 2:
            return False, ""

        # Проверка кд
        if current_time - self.last_alert_time < config.cooldown_sec:
            return False, ""

        alerts = []

        # Относительные изменения (5 секунд)
        target_time = current_time - 5
        ref_price = self._get_reference_price(target_time)

        if ref_price and ref_price > 0:
            change_pct = (current_price - ref_price) / ref_price * 100

            if change_pct >= config.threshold_up:
                alerts.append(f"▲ +{change_pct:.2f}%")
            elif change_pct <= -config.threshold_down:
                alerts.append(f"▼ {change_pct:.2f}%")

        # Абсолютные пороги
        if config.absolute_up is not None and current_price >= config.absolute_up:
            alerts.append(f"Цена ≥ {config.absolute_up}")
        if config.absolute_down is not None and current_price <= config.absolute_down:
            alerts.append(f"Цена ≤ {config.absolute_down}")

        if alerts:
            self.last_alert_time = current_time
            return True, " | ".join(alerts)

        return False, ""

    def _get_reference_price(self, target_time: float) -> Optional[float]:
        """Получает цену за указанное время."""
        for timestamp, price in reversed(self.price_history):
            if timestamp <= target_time:
                return price
        return None


class CurrencyConverter:
    """Конвертер валют."""

    def __init__(self):
        self.rates: Dict[str, float] = {"USD": 1.0, "RUB": 95.0}
        self.last_update = 0

    def update_rates(self) -> bool:
        """Обновляет курсы валют."""
        try:
            import requests
            resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=USDTRUB", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                usdt_rub = float(data['price'])
                self.rates["RUB"] = usdt_rub
                self.last_update = time.time()
                return True
        except Exception as e:
            print(f"Currency update error: {e}")
        return False

    def get_usdt_rub(self) -> float:
        """Получает курс USDT/RUB."""
        if time.time() - self.last_update > 3600:  # Обновлять раз в час
            self.update_rates()
        return self.rates.get("RUB", 95.0)

    def convert(self, amount: float, from_curr: str, to_curr: str) -> Optional[float]:
        """Конвертирует сумму между валютами."""
        try:
            if from_curr == to_curr:
                return amount

            # Простая конвертация через USD
            if from_curr != "USD":
                amount = amount / self.rates.get(from_curr, 1.0)

            if to_curr != "USD":
                amount = amount * self.rates.get(to_curr, 1.0)

            return amount
        except Exception:
            return None