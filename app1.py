import tkinter as tk
from tkinter import font, ttk, messagebox
import threading
import json
import websocket
import time
import os
import sys
import logging
import math
from datetime import datetime
from collections import deque
import locale
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Настройка логирования
logging.basicConfig(
    filename='crypto_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Попытка импорта внешних библиотек
try:
    import winsound

    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    from pystray import Icon, Menu, MenuItem
    from PIL import Image, ImageDraw

    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False
    logging.warning("pystray not available")

try:
    from plyer import notification

    HAS_NOTIFICATIONS = True
except ImportError:
    HAS_NOTIFICATIONS = False
    logging.warning("plyer not available")

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Глобальные константы
CONFIG_FILE = "config.json"
LOCALES_DIR = "locales"
MAX_HISTORY_POINTS = 120
MAX_GUI_UPDATES_PER_SEC = 2
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/"
BINANCE_REST_URL = "https://api.binance.com/api/v3/ticker/price?symbol="

# Список популярных пар
POPULAR_SYMBOLS = [
    "btcusdt", "ethusdt", "solusdt",
    "btceur", "etheur", "soleur",
    "btcrub", "ethrub", "solrub"
]


class AIPredictor:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def create_features(self, prices):
        """Создание фич для ML модели"""
        if len(prices) < self.window_size:
            return None

        features = []
        # Простые скользящие средние
        for window in [5, 10, 15]:
            if len(prices) >= window:
                features.append(np.mean(prices[-window:]))

        # Волатильность
        returns = np.diff(prices[-10:]) / prices[-11:-1]
        if len(returns) > 0:
            features.append(np.std(returns))

        # RSI-like feature
        gains = [max(0, prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        losses = [max(0, prices[i - 1] - prices[i]) for i in range(1, len(prices))]
        if sum(losses) > 0:
            rs = sum(gains[-14:]) / sum(losses[-14:]) if len(gains) >= 14 else 1
            features.append(100 - (100 / (1 + rs)))
        else:
            features.append(100)

        return np.array(features).reshape(1, -1)

    def train(self, prices):
        """Обучение модели на исторических данных"""
        if len(prices) < self.window_size + 10:
            return False

        X, y = [], []
        for i in range(self.window_size, len(prices) - 1):
            feature_window = prices[i - self.window_size:i]
            features = self.create_features(feature_window)
            if features is not None:
                X.append(features.flatten())
                # Целевая переменная - изменение цены в следующем периоде
                price_change = (prices[i + 1] - prices[i]) / prices[i] * 100
                y.append(price_change)

        if len(X) > 5:
            X = np.array(X)
            y = np.array(y)

            # Масштабирование фич
            X_scaled = self.scaler.fit_transform(X)

            # Обучение модели
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logging.info("AI model trained successfully")
            return True
        return False

    def predict(self, prices):
        """Предсказание изменения цены"""
        if not self.is_trained or len(prices) < self.window_size:
            return None, 0.0

        features = self.create_features(prices)
        if features is None:
            return None, 0.0

        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]

        # Доверительный интервал (простая оценка)
        confidence = min(0.95, max(0.1, 1 - abs(prediction) / 10))

        return prediction, confidence


class TechnicalAnalyzer:
    def __init__(self, period=14):
        self.period = period

    def calculate_rsi(self, prices):
        """Расчет RSI (Relative Strength Index)"""
        if len(prices) < self.period + 1:
            return 50.0  # Нейтральное значение при недостатке данных

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.convolve(gains, np.ones(self.period) / self.period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(self.period) / self.period, mode='valid')

        # Избегаем деления на ноль
        avg_losses = np.where(avg_losses == 0, 1e-10, avg_losses)

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return float(rsi[-1]) if len(rsi) > 0 else 50.0

    def calculate_volume_profile(self, volumes, prices, num_bins=10):
        """Анализ профиля объема"""
        if len(volumes) < 2 or len(prices) < 2:
            return None, None

        # Простой анализ объема
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        return volume_ratio, avg_volume

    def calculate_support_resistance(self, prices, window=20):
        """Определение уровней поддержки и сопротивления"""
        if len(prices) < window:
            return None, None

        recent_prices = prices[-window:]
        support = np.min(recent_prices)
        resistance = np.max(recent_prices)

        return support, resistance


class AdvancedAlertManager:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.ai_predictor = AIPredictor()
        self.alert_history = []
        self.volume_history = deque(maxlen=50)

    def check_technical_alerts(self, current_price, prices, volumes, config):
        """Проверка технических алертов"""
        alerts = []

        # RSI алерты
        rsi = self.technical_analyzer.calculate_rsi(prices)
        rsi_overbought = config.get("rsi_overbought", 70)
        rsi_oversold = config.get("rsi_oversold", 30)

        if rsi >= rsi_overbought:
            alerts.append(f"RSI ПЕРЕКУП: {rsi:.1f}")
        elif rsi <= rsi_oversold:
            alerts.append(f"RSI ПРОДАЖА: {rsi:.1f}")

        # Volume алерты
        if volumes and len(volumes) > 5:
            volume_ratio, avg_volume = self.technical_analyzer.calculate_volume_profile(volumes, prices)
            volume_threshold = config.get("volume_threshold", 2.0)

            if volume_ratio and volume_ratio >= volume_threshold:
                alerts.append(f"ОБЪЕМ x{volume_ratio:.1f}")

        # Support/Resistance алерты
        support, resistance = self.technical_analyzer.calculate_support_resistance(prices)
        if support and resistance:
            support_break_threshold = config.get("support_break_threshold", 0.98)  # 2% ниже поддержки
            resistance_break_threshold = config.get("resistance_break_threshold", 1.02)  # 2% выше сопротивления

            if current_price <= support * support_break_threshold:
                alerts.append(f"ПРОБИТИЕ ПОДДЕРЖКИ: {support:.2f}")
            elif current_price >= resistance * resistance_break_threshold:
                alerts.append(f"ПРОБИТИЕ СОПРОТИВЛЕНИЯ: {resistance:.2f}")

        return alerts

    def check_ai_alerts(self, prices, config):
        """Проверка AI предсказаний"""
        alerts = []

        if len(prices) >= 30:  # Минимум данных для предсказания
            # Переобучаем модель периодически
            if len(prices) % 50 == 0:  # Каждые 50 новых точек
                self.ai_predictor.train(prices)

            prediction, confidence = self.ai_predictor.predict(prices)

            if prediction is not None:
                prediction_threshold = config.get("prediction_threshold", 2.0)  # 2% изменение
                confidence_threshold = config.get("confidence_threshold", 0.7)  # 70% уверенность

                if abs(prediction) >= prediction_threshold and confidence >= confidence_threshold:
                    direction = "РОСТ" if prediction > 0 else "ПАДЕНИЕ"
                    alerts.append(f"AI: {direction} {abs(prediction):.1f}% (уверенность: {confidence:.0%})")

        return alerts


class PriceMonitor:
    def __init__(self, symbol="btcusdt"):
        self.symbol = symbol.lower()
        self.running = True
        self.price_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.volume_history = deque(maxlen=MAX_HISTORY_POINTS)
        self.last_alert_time = 0
        self.last_price = None
        self.on_price_update = None
        self.on_status_change = None
        self.advanced_alerts = AdvancedAlertManager()

    def set_callbacks(self, on_price, on_status):
        self.on_price_update = on_price
        self.on_status_change = on_status

    def add_price(self, price, volume=0):
        if not self.running or price <= 0 or not math.isfinite(price):
            return
        now = time.time()
        self.price_history.append((now, price))
        if volume > 0:
            self.volume_history.append((now, volume))
        if self.on_price_update:
            self.on_price_update(price, now)

    def check_alert(self, current_price, current_time, config):
        if not self.price_history or len(self.price_history) < 2:
            return False, ""

        cooldown = config.get("alert_cooldown_sec", 15)
        if current_time - self.last_alert_time < cooldown:
            return False, ""

        alerts = []

        # Базовые алерты (изменение цены)
        target_time = current_time - 5
        ref_price = None
        for t, p in reversed(self.price_history):
            if t <= target_time:
                ref_price = p
                break

        if ref_price and ref_price > 0:
            change_pct = (current_price - ref_price) / ref_price * 100
            up_threshold = config.get("alert_threshold_up", 1.0)
            down_threshold = config.get("alert_threshold_down", 1.0)

            if change_pct >= up_threshold:
                alerts.append(f"▲ +{change_pct:.2f}%")
            elif change_pct <= -down_threshold:
                alerts.append(f"▼ {change_pct:.2f}%")

        # Абсолютные пороги
        abs_up = config.get("alert_absolute_up")
        abs_down = config.get("alert_absolute_down")
        if abs_up is not None and current_price >= abs_up:
            alerts.append(f"Цена ≥ {abs_up}")
        if abs_down is not None and current_price <= abs_down:
            alerts.append(f"Цена ≤ {abs_down}")

        # Расширенные технические алерты
        if config.get("enable_technical_alerts", False):
            prices = [p for _, p in self.price_history]
            volumes = [v for _, v in self.volume_history] if self.volume_history else []
            technical_alerts = self.advanced_alerts.check_technical_alerts(current_price, prices, volumes, config)
            alerts.extend(technical_alerts)

        # AI предсказания
        if config.get("enable_ai_predictions", False):
            prices = [p for _, p in self.price_history]
            ai_alerts = self.advanced_alerts.check_ai_alerts(prices, config)
            alerts.extend(ai_alerts)

        if alerts:
            self.last_alert_time = current_time
            return True, " | ".join(alerts)

        return False, ""


class CurrencyConverter:
    def __init__(self):
        self.rates = {"USD": 1.0, "RUB": 95.0}  # Начальное значение для RUB
        self.last_update = 0

    def update_rates(self):
        """Обновление курсов валют"""
        if not HAS_REQUESTS:
            return

        try:
            # Получаем курс USDT/RUB через Binance
            resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=USDTRUB", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                usdt_rub = float(data['price'])
                self.rates["RUB"] = usdt_rub
                self.last_update = time.time()
                logging.info(f"Updated USDT/RUB rate: {usdt_rub}")
        except Exception as e:
            logging.error(f"Currency update error: {e}")
            # Если не удалось получить актуальный курс, используем последний известный

    def get_rate(self, to_currency):
        if to_currency == "USD":
            return 1.0

        # Обновляем курсы раз в час
        if time.time() - self.last_update > 3600:
            self.update_rates()

        return self.rates.get(to_currency, 1.0)

    def get_usdt_rub(self):
        """Получить актуальный курс USDT/RUB"""
        if time.time() - self.last_update > 3600:
            self.update_rates()
        return self.rates.get("RUB", 95.0)


class AlertHistoryWindow:
    def __init__(self, parent, alerts):
        self.window = tk.Toplevel(parent)
        self.window.title("История оповещений")
        self.window.geometry("520x320")
        self.window.configure(bg="#2b2b2b")

        frame = tk.Frame(self.window, bg="#2b2b2b")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            frame, yscrollcommand=scrollbar.set,
            bg="#333", fg="white", font=("Consolas", 10),
            width=80
        )
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        for alert in reversed(alerts):
            self.listbox.insert("end", alert)


class ConverterWindow:
    def __init__(self, parent, converter, price_monitor, translations):
        self.window = tk.Toplevel(parent)
        self.window.title(translations.get("converter_title", "Конвертер"))
        self.window.geometry("320x200")
        self.window.configure(bg="#2b2b2b")
        self.converter = converter
        self.price_monitor = price_monitor
        self._ = lambda k: translations.get(k, k)

        main_frame = tk.Frame(self.window, bg="#2b2b2b")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        tk.Label(main_frame, text=self._("enter_amount"), bg="#2b2b2b", fg="white", font=("Arial", 10)).pack(anchor="w",
                                                                                                             pady=(0,
                                                                                                                   8))

        # Поле ввода
        input_frame = tk.Frame(main_frame, bg="#2b2b2b")
        input_frame.pack(fill="x", pady=5)

        self.amount_var = tk.StringVar()
        self.amount_var.trace("w", self.convert)
        tk.Entry(input_frame, textvariable=self.amount_var, width=20, font=("Arial", 11)).pack(side="left",
                                                                                               padx=(0, 10))

        # Переключатель режима
        mode_frame = tk.Frame(main_frame, bg="#2b2b2b")
        mode_frame.pack(fill="x", pady=8)

        self.mode_var = tk.StringVar(value="btc")
        tk.Radiobutton(mode_frame, text="BTC → RUB", variable=self.mode_var, value="btc",
                       bg="#2b2b2b", fg="white", selectcolor="#444",
                       font=("Arial", 9), command=self.convert).pack(side="left", padx=(0, 15))
        tk.Radiobutton(mode_frame, text="USDT → RUB", variable=self.mode_var, value="usdt",
                       bg="#2b2b2b", fg="white", selectcolor="#444",
                       font=("Arial", 9), command=self.convert).pack(side="left")

        # Результат
        result_frame = tk.Frame(main_frame, bg="#2b2b2b")
        result_frame.pack(fill="x", pady=15)

        tk.Label(result_frame, text="Результат:", bg="#2b2b2b", fg="white", font=("Arial", 10)).pack(anchor="w")
        self.result_label = tk.Label(result_frame, text="", bg="#2b2b2b", fg="lightgreen",
                                     font=("Consolas", 12, "bold"))
        self.result_label.pack(anchor="w", pady=(5, 0))

        # Информация о курсах
        info_frame = tk.Frame(main_frame, bg="#2b2b2b")
        info_frame.pack(fill="x", pady=(10, 0))

        self.rate_info = tk.Label(info_frame, text="", bg="#2b2b2b", fg="#888",
                                  font=("Arial", 8))
        self.rate_info.pack(anchor="w")

        # Обновляем информацию о курсах
        self.update_rate_info()

    def update_rate_info(self):
        """Обновить информацию о текущих курсах"""
        btc_price = self.price_monitor.last_price or 0
        usdt_rub = self.converter.get_usdt_rub()

        if btc_price > 0:
            self.rate_info.config(text=f"BTC: ${btc_price:,.2f} | USDT/RUB: {usdt_rub:.2f}")
        else:
            self.rate_info.config(text=f"USDT/RUB: {usdt_rub:.2f}")

    def convert(self, *args):
        try:
            amount_text = self.amount_var.get().strip()
            if not amount_text:
                self.result_label.config(text="")
                return

            amount = float(amount_text)
            mode = self.mode_var.get()

            # Обновляем информацию о курсах
            self.update_rate_info()

            if mode == "btc":
                # Конвертация BTC → RUB
                btc_price_usd = self.price_monitor.last_price if self.price_monitor.last_price else 0
                if btc_price_usd <= 0:
                    self.result_label.config(text="Нет данных о цене BTC")
                    return

                usdt_rub_rate = self.converter.get_usdt_rub()
                rub_amount = amount * btc_price_usd * usdt_rub_rate
                self.result_label.config(text=f"≈ {rub_amount:,.0f} RUB")

            else:  # usdt → rub
                # Конвертация USDT → RUB
                usdt_rub_rate = self.converter.get_usdt_rub()
                rub_amount = amount * usdt_rub_rate
                self.result_label.config(text=f"≈ {rub_amount:,.0f} RUB")

        except (ValueError, TypeError):
            self.result_label.config(text="Введите число")


class AdvancedSettingsWindow:
    def __init__(self, parent, config, save_callback, translations):
        self.window = tk.Toplevel(parent)
        self.window.title("Расширенные настройки алертов")
        self.window.geometry("480x600")
        self.window.resizable(False, False)
        self.window.configure(bg="#2b2b2b")
        self.window.transient(parent)
        self.window.grab_set()

        self.config = config.copy()
        self.save_callback = save_callback
        self._ = lambda k: translations.get(k, k)

        # Canvas + Scrollbar
        canvas = tk.Canvas(self.window, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.window.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        content = scrollable_frame

        # Технические алерты
        tk.Label(content, text="Технические алерты", bg="#2b2b2b", fg="#FFD700",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(10, 5), anchor="w")

        self.tech_alerts_var = tk.BooleanVar(value=config.get("enable_technical_alerts", False))
        tk.Checkbutton(content, text="Включить технические алерты", variable=self.tech_alerts_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20, pady=(5, 0))

        # RSI настройки
        tk.Label(content, text="RSI Перекупленность (>):", bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.rsi_overbought_var = tk.DoubleVar(value=config.get("rsi_overbought", 70))
        ttk.Scale(content, from_=60, to=90, variable=self.rsi_overbought_var, orient="horizontal").pack(fill="x",
                                                                                                        padx=20)
        self.rsi_overbought_label = tk.Label(content, text=f"{self.rsi_overbought_var.get():.0f}", bg="#2b2b2b",
                                             fg="#aaa")
        self.rsi_overbought_label.pack()
        self.rsi_overbought_var.trace("w", lambda *a: self.rsi_overbought_label.config(
            text=f"{self.rsi_overbought_var.get():.0f}"))

        tk.Label(content, text="RSI Перепроданность (<):", bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        self.rsi_oversold_var = tk.DoubleVar(value=config.get("rsi_oversold", 30))
        ttk.Scale(content, from_=10, to=40, variable=self.rsi_oversold_var, orient="horizontal").pack(fill="x", padx=20)
        self.rsi_oversold_label = tk.Label(content, text=f"{self.rsi_oversold_var.get():.0f}", bg="#2b2b2b", fg="#aaa")
        self.rsi_oversold_label.pack()
        self.rsi_oversold_var.trace("w", lambda *a: self.rsi_oversold_label.config(
            text=f"{self.rsi_oversold_var.get():.0f}"))

        # Volume настройки
        tk.Label(content, text="Порог объема (x от среднего):", bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.volume_threshold_var = tk.DoubleVar(value=config.get("volume_threshold", 2.0))
        ttk.Scale(content, from_=1.0, to=5.0, variable=self.volume_threshold_var, orient="horizontal").pack(fill="x",
                                                                                                            padx=20)
        self.volume_threshold_label = tk.Label(content, text=f"x{self.volume_threshold_var.get():.1f}", bg="#2b2b2b",
                                               fg="#aaa")
        self.volume_threshold_label.pack()
        self.volume_threshold_var.trace("w", lambda *a: self.volume_threshold_label.config(
            text=f"x{self.volume_threshold_var.get():.1f}"))

        # AI предсказания
        tk.Label(content, text="AI предсказания", bg="#2b2b2b", fg="#FFD700",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(15, 5), anchor="w")

        self.ai_predictions_var = tk.BooleanVar(value=config.get("enable_ai_predictions", False))
        tk.Checkbutton(content, text="Включить AI предсказания", variable=self.ai_predictions_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20, pady=(5, 0))

        tk.Label(content, text="Порог предсказания (% изменение):", bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.prediction_threshold_var = tk.DoubleVar(value=config.get("prediction_threshold", 2.0))
        ttk.Scale(content, from_=0.5, to=5.0, variable=self.prediction_threshold_var, orient="horizontal").pack(
            fill="x", padx=20)
        self.prediction_threshold_label = tk.Label(content, text=f"{self.prediction_threshold_var.get():.1f}%",
                                                   bg="#2b2b2b", fg="#aaa")
        self.prediction_threshold_label.pack()
        self.prediction_threshold_var.trace("w", lambda *a: self.prediction_threshold_label.config(
            text=f"{self.prediction_threshold_var.get():.1f}%"))

        tk.Label(content, text="Минимальная уверенность AI:", bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        self.confidence_threshold_var = tk.DoubleVar(value=config.get("confidence_threshold", 0.7))
        ttk.Scale(content, from_=0.1, to=0.95, variable=self.confidence_threshold_var, orient="horizontal").pack(
            fill="x", padx=20)
        self.confidence_threshold_label = tk.Label(content, text=f"{self.confidence_threshold_var.get():.0%}",
                                                   bg="#2b2b2b", fg="#aaa")
        self.confidence_threshold_label.pack()
        self.confidence_threshold_var.trace("w", lambda *a: self.confidence_threshold_label.config(
            text=f"{self.confidence_threshold_var.get():.0%}"))

        # Support/Resistance
        tk.Label(content, text="Уровни поддержки/сопротивления", bg="#2b2b2b", fg="#FFD700",
                 font=("TkDefaultFont", 11, "bold")).pack(pady=(15, 5), anchor="w")

        tk.Label(content, text="Порог пробоя поддержки (% ниже):", bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        self.support_break_var = tk.DoubleVar(value=config.get("support_break_threshold", 0.98))
        ttk.Scale(content, from_=0.95, to=1.0, variable=self.support_break_var, orient="horizontal").pack(fill="x",
                                                                                                          padx=20)
        self.support_break_label = tk.Label(content, text=f"{self.support_break_var.get():.1%}", bg="#2b2b2b",
                                            fg="#aaa")
        self.support_break_label.pack()
        self.support_break_var.trace("w", lambda *a: self.support_break_label.config(
            text=f"{self.support_break_var.get():.1%}"))

        tk.Label(content, text="Порог пробоя сопротивления (% выше):", bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        self.resistance_break_var = tk.DoubleVar(value=config.get("resistance_break_threshold", 1.02))
        ttk.Scale(content, from_=1.0, to=1.05, variable=self.resistance_break_var, orient="horizontal").pack(fill="x",
                                                                                                             padx=20)
        self.resistance_break_label = tk.Label(content, text=f"{self.resistance_break_var.get():.1%}", bg="#2b2b2b",
                                               fg="#aaa")
        self.resistance_break_label.pack()
        self.resistance_break_var.trace("w", lambda *a: self.resistance_break_label.config(
            text=f"{self.resistance_break_var.get():.1%}"))

        # Кнопка сохранения
        tk.Button(content, text="Сохранить", command=self.save,
                  bg="#444", fg="white", relief="flat", width=15).pack(pady=20)

    def save(self):
        self.config.update({
            "enable_technical_alerts": self.tech_alerts_var.get(),
            "rsi_overbought": self.rsi_overbought_var.get(),
            "rsi_oversold": self.rsi_oversold_var.get(),
            "volume_threshold": self.volume_threshold_var.get(),
            "enable_ai_predictions": self.ai_predictions_var.get(),
            "prediction_threshold": self.prediction_threshold_var.get(),
            "confidence_threshold": self.confidence_threshold_var.get(),
            "support_break_threshold": self.support_break_var.get(),
            "resistance_break_threshold": self.resistance_break_var.get()
        })
        self.save_callback(self.config)
        self.window.destroy()


class SettingsWindow:
    def __init__(self, parent, config, save_callback, translations):
        self.window = tk.Toplevel(parent)
        self.window.title(translations.get("settings_title", "Settings"))
        self.window.geometry("440x500")
        self.window.resizable(False, False)
        self.window.configure(bg="#2b2b2b")
        self.window.transient(parent)
        self.window.grab_set()

        self.config = config.copy()
        self.save_callback = save_callback
        self._ = lambda k: translations.get(k, k)

        # Canvas + Scrollbar
        canvas = tk.Canvas(self.window, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2b2b2b")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.window.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        content = scrollable_frame

        # Язык
        tk.Label(content, text=self._("language"), bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.lang_var = tk.StringVar(value=config.get("language", "auto"))
        langs = [("Авто", "auto"), ("Русский", "ru"), ("English", "en")]
        for text, val in langs:
            tk.Radiobutton(content, text=text, variable=self.lang_var, value=val,
                           bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20)

        # Пара
        tk.Label(content, text=self._("trading_pair"), bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.symbol_var = tk.StringVar(value=config.get("symbol", "btcusdt"))
        symbol_combo = ttk.Combobox(content, textvariable=self.symbol_var, values=POPULAR_SYMBOLS, state="readonly",
                                    width=20)
        symbol_combo.pack(padx=20)

        # Относительные пороги
        tk.Label(content, text=self._("alert_threshold_up"), bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        self.up_var = tk.DoubleVar(value=config.get("alert_threshold_up", 1.0))
        ttk.Scale(content, from_=0.1, to=10.0, variable=self.up_var, orient="horizontal").pack(fill="x", padx=20)
        self.up_label = tk.Label(content, text=f"{self.up_var.get():.1f}%", bg="#2b2b2b", fg="#aaa")
        self.up_label.pack()
        self.up_var.trace("w", lambda *a: self.up_label.config(text=f"{self.up_var.get():.1f}%"))

        tk.Label(content, text=self._("alert_threshold_down"), bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        self.down_var = tk.DoubleVar(value=config.get("alert_threshold_down", 1.0))
        ttk.Scale(content, from_=0.1, to=10.0, variable=self.down_var, orient="horizontal").pack(fill="x", padx=20)
        self.down_label = tk.Label(content, text=f"{self.down_var.get():.1f}%", bg="#2b2b2b", fg="#aaa")
        self.down_label.pack()
        self.down_var.trace("w", lambda *a: self.down_label.config(text=f"{self.down_var.get():.1f}%"))

        # Абсолютные пороги
        tk.Label(content, text=self._("alert_absolute_up"), bg="#2b2b2b", fg="white").pack(pady=(10, 0))
        abs_up_val = config.get("alert_absolute_up")
        self.abs_up_var = tk.StringVar(value=str(abs_up_val) if abs_up_val is not None else "")
        tk.Entry(content, textvariable=self.abs_up_var, width=15).pack()

        tk.Label(content, text=self._("alert_absolute_down"), bg="#2b2b2b", fg="white").pack(pady=(5, 0))
        abs_down_val = config.get("alert_absolute_down")
        self.abs_down_var = tk.StringVar(value=str(abs_down_val) if abs_down_val is not None else "")
        tk.Entry(content, textvariable=self.abs_down_var, width=15).pack()

        # Звук и уведомления
        self.sound_var = tk.BooleanVar(value=config.get("show_sound", True))
        tk.Checkbutton(content, text=self._("sound_enabled"), variable=self.sound_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20, pady=(10, 0))

        self.notify_var = tk.BooleanVar(value=config.get("show_notifications", True))
        tk.Checkbutton(content, text=self._("notifications_enabled"), variable=self.notify_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20)

        # Отображение
        tk.Label(content, text=self._("display_options"), bg="#2b2b2b", fg="white",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=(15, 5), anchor="w", padx=20)

        self.show_chart_var = tk.BooleanVar(value=config.get("show_chart", True))
        tk.Checkbutton(content, text=self._("show_chart"), variable=self.show_chart_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20)

        self.color_price_var = tk.BooleanVar(value=config.get("color_price", True))
        tk.Checkbutton(content, text=self._("color_price"), variable=self.color_price_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w", padx=20)

        # Кнопка расширенных настроек
        tk.Button(content, text="Расширенные алерты", command=self.open_advanced_settings,
                  bg="#555", fg="white", relief="flat", width=20).pack(pady=10)

        # Кнопка сохранения
        tk.Button(content, text=self._("save"), command=self.save,
                  bg="#444", fg="white", relief="flat", width=15).pack(pady=10)

    def open_advanced_settings(self):
        AdvancedSettingsWindow(self.window, self.config, self.save_callback, self.translations)

    def save(self):
        try:
            abs_up_str = self.abs_up_var.get().strip()
            abs_down_str = self.abs_down_var.get().strip()
            abs_up = float(abs_up_str) if abs_up_str else None
            abs_down = float(abs_down_str) if abs_down_str else None
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные абсолютные пороги")
            return

        self.config.update({
            "language": self.lang_var.get(),
            "symbol": self.symbol_var.get().lower(),
            "alert_threshold_up": self.up_var.get(),
            "alert_threshold_down": self.down_var.get(),
            "alert_absolute_up": abs_up,
            "alert_absolute_down": abs_down,
            "show_sound": self.sound_var.get(),
            "show_notifications": self.notify_var.get(),
            "show_chart": self.show_chart_var.get(),
            "color_price": self.color_price_var.get()
        })
        self.save_callback(self.config)
        self.window.destroy()


class CryptoPriceApp:
    def __init__(self, root):
        self.root = root
        self.alert_history = deque(maxlen=50)
        self.last_chart_draw = 0
        self.chart_throttle_sec = 0.5

        # Определение языка
        sys_lang = locale.getdefaultlocale()[0]
        default_lang = "ru" if sys_lang and sys_lang.startswith("ru") else "en"

        self.config = self.load_config()
        if self.config.get("language") == "auto":
            self.config["language"] = default_lang

        self.translations = self.load_translations(self.config["language"])

        self.root.title(self._("title"))
        self.root.geometry("520x420")
        self.root.resizable(False, False)
        self.apply_theme()

        if self.config.get("always_on_top", False):
            self.root.wm_attributes("-topmost", True)

        # DPI scaling (для Windows)
        if sys.platform == "win32":
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass

        self.set_icon()

        self.converter = CurrencyConverter()
        self.monitor = PriceMonitor(self.config["symbol"])
        self.monitor.set_callbacks(self.on_price_update, self.on_status_update)

        self.chart_history = deque(maxlen=120)
        self.setup_ui()
        self.setup_tray()

        self.last_gui_update = 0
        self.ws_thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.ws_thread.start()
        self.rest_fallback_thread = threading.Thread(target=self.rest_fallback_loop, daemon=True)
        self.rest_fallback_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind('<Control-q>', lambda e: self.quit_app())
        self.root.bind('<Control-h>', lambda e: self.show_window())
        self.root.bind('<Control-l>', lambda e: self.show_alert_history())
        self.root.bind('<Control-a>', lambda e: self.open_advanced_settings())

        if self.config.get("start_minimized", False) and HAS_TRAY:
            self.hide_to_tray()

    def _(self, key):
        return self.translations.get(key, key)

    def load_config(self):
        default_config = {
            "language": "auto",
            "symbol": "btcusdt",
            "alert_threshold_up": 1.0,
            "alert_threshold_down": 1.0,
            "alert_cooldown_sec": 15,
            "start_minimized": False,
            "always_on_top": False,
            "show_notifications": True,
            "show_sound": True,
            "show_chart": True,
            "color_price": True,
            # Расширенные настройки
            "enable_technical_alerts": False,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "volume_threshold": 2.0,
            "enable_ai_predictions": False,
            "prediction_threshold": 2.0,
            "confidence_threshold": 0.7,
            "support_break_threshold": 0.98,
            "resistance_break_threshold": 1.02
        }

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for key, value in default_config.items():
                    if key not in loaded:
                        loaded[key] = value
                return loaded
            except Exception as e:
                logging.error(f"Config load error: {e}")

        return default_config

    def save_config(self, new_config):
        self.config.update(new_config)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.update_title()
            self.toggle_chart_visibility()
        except Exception as e:
            logging.error(f"Config save error: {e}")

    def load_translations(self, lang):
        path = os.path.join(LOCALES_DIR, f"{lang}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
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
            "alert_absolute_up": "Alert if price ≥ (e.g. 70000)",
            "alert_absolute_down": "Alert if price ≤ (e.g. 60000)",
            "sound_enabled": "Sound Alert",
            "notifications_enabled": "System Notifications",
            "display_options": "Display Options",
            "show_chart": "Show price chart",
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
            "enter_amount": "Enter amount (BTC or USDT):"
        }

    def apply_theme(self):
        # Фиксированная тёмная тема
        bg = "#1e1e1e"
        fg = "white"
        btn_bg = "#333"

        self.root.configure(bg=bg)
        if hasattr(self, 'title_label'):
            self.title_label.config(bg=bg, fg="#FFD700")
        if hasattr(self, 'price_label'):
            self.price_label.config(bg=bg)
        if hasattr(self, 'converted_label'):
            self.converted_label.config(bg=bg, fg="#aaa")
        if hasattr(self, 'usdt_rub_label'):
            self.usdt_rub_label.config(bg=bg, fg="#888")
        if hasattr(self, 'status_label'):
            self.status_label.config(bg=bg, fg="#888")
        if hasattr(self, 'graph_frame'):
            self.graph_frame.config(bg=bg)
        if hasattr(self, 'chart_canvas'):
            chart_bg = "#2a2a2a"
            self.chart_canvas.config(bg=chart_bg)
        if hasattr(self, 'always_on_top_btn'):
            self.always_on_top_btn.config(bg=btn_bg, fg=fg)

    def set_icon(self):
        try:
            from PIL import Image, ImageDraw, ImageTk
            img = Image.new("RGBA", (16, 16), (30, 30, 30, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([4, 4, 11, 11], outline=(255, 215, 0), width=1)
            draw.text((5, 2), "C", fill=(255, 215, 0))
            icon = ImageTk.PhotoImage(img)
            self.root.iconphoto(True, icon)
        except:
            pass

    def update_title(self):
        symbol = self.config["symbol"].upper()
        base = symbol[-3:].upper()
        pair = f"{symbol[:-3].upper()}/{base}"
        self.title_label.config(text=pair)

    def toggle_chart_visibility(self):
        if self.config.get("show_chart", True):
            self.graph_frame.pack(fill="x", padx=20, pady=5)
        else:
            self.graph_frame.pack_forget()

    def setup_ui(self):
        self.title_label = tk.Label(
            self.root, text="", font=font.Font(size=18, weight="bold"),
            bg="#1e1e1e", fg="#FFD700"
        )
        self.update_title()
        self.title_label.pack(pady=8)

        self.price_label = tk.Label(
            self.root, text="Loading...", font=font.Font(family="Consolas", size=24, weight="bold"),
            bg="#1e1e1e", fg="white"
        )
        self.price_label.pack()

        self.converted_label = tk.Label(
            self.root, text="", font=("Consolas", 14), bg="#1e1e1e", fg="#aaa"
        )
        self.converted_label.pack(pady=(0, 5))

        self.graph_frame = tk.Frame(self.root, height=100, bg="#1e1e1e")
        self.graph_frame.pack_propagate(False)
        self.toggle_chart_visibility()

        self.chart_canvas = tk.Canvas(self.graph_frame, bg="#2a2a2a", highlightthickness=0, height=100)
        self.chart_canvas.pack(fill="both", expand=True)
        self.chart_canvas.bind("<Configure>", self.on_chart_resize)

        # USDT/RUB label
        usdt_rub = self.converter.get_usdt_rub()
        self.usdt_rub_label = tk.Label(
            self.root, text=f"USDT/RUB ≈ ₽{usdt_rub:,.2f}",
            font=("Consolas", 10), bg="#1e1e1e", fg="#888"
        )
        self.usdt_rub_label.pack(pady=(2, 0))

        status_frame = tk.Frame(self.root, bg="#1e1e1e")
        status_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        self.status_indicator = tk.Label(status_frame, width=2, bg="orange")
        self.status_indicator.pack(side="left", padx=(0, 5))
        self.status_label = tk.Label(status_frame, text=self._("connecting"), fg="#888", bg="#1e1e1e")
        self.status_label.pack(side="left")

        # Контейнер для кнопок в 2 ряда
        btn_container = tk.Frame(self.root, bg="#1e1e1e")
        btn_container.pack(side="bottom", pady=8)

        # Первый ряд кнопок (3 кнопки)
        btn_row1 = tk.Frame(btn_container, bg="#1e1e1e")
        btn_row1.pack(pady=2)

        if HAS_TRAY:
            btn_min = tk.Button(btn_row1, text=self._("btn_minimize"), command=self.hide_to_tray,
                                bg="#333", fg="white", relief="flat", width=14)
            btn_min.pack(side="left", padx=2)
            self.create_tooltip(btn_min, "Скрыть в системный трей (Ctrl+H для показа)")

        self.always_on_top_btn = tk.Button(
            btn_row1, text=self._("btn_always_on_top"),
            command=self.toggle_always_on_top,
            bg="#333" if self.config["always_on_top"] else "#444",
            fg="white", relief="flat", width=14
        )
        self.always_on_top_btn.pack(side="left", padx=2)
        self.create_tooltip(self.always_on_top_btn, "Поверх всех окон")

        btn_set = tk.Button(btn_row1, text=self._("btn_settings"), command=self.open_settings,
                            bg="#333", fg="white", relief="flat", width=14)
        btn_set.pack(side="left", padx=2)
        self.create_tooltip(btn_set, "Настройки приложения (Ctrl+Q — выход)")

        # Второй ряд кнопок (3 кнопки)
        btn_row2 = tk.Frame(btn_container, bg="#1e1e1e")
        btn_row2.pack(pady=2)

        btn_hist = tk.Button(btn_row2, text="История", command=self.show_alert_history,
                             bg="#333", fg="white", relief="flat", width=14)
        btn_hist.pack(side="left", padx=2)
        self.create_tooltip(btn_hist, "Показать историю оповещений (Ctrl+L)")

        btn_conv = tk.Button(btn_row2, text="Конвертер", command=self.open_converter,
                             bg="#333", fg="white", relief="flat", width=14)
        btn_conv.pack(side="left", padx=2)
        self.create_tooltip(btn_conv, "Конвертер BTC/USDT → RUB")

        btn_adv = tk.Button(btn_row2, text="AI Алерты", command=self.open_advanced_settings,
                            bg="#555", fg="white", relief="flat", width=14)
        btn_adv.pack(side="left", padx=2)
        self.create_tooltip(btn_adv, "Расширенные алерты (Ctrl+A)")

    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="#333", fg="white", relief="solid", borderwidth=1)
        label.pack()

        def show(e):
            x = e.widget.winfo_rootx() + 20
            y = e.widget.winfo_rooty() + 20
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def hide(e):
            tooltip.withdraw()

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def setup_tray(self):
        if not HAS_TRAY:
            return

        def create_image():
            image = Image.new('RGB', (16, 16), (30, 30, 30))
            draw = ImageDraw.Draw(image)
            draw.rectangle([4, 4, 11, 11], outline=(255, 215, 0))
            draw.text((5, 2), "C", fill=(255, 215, 0))
            return image

        def show_window(icon, item=None):
            icon.stop()
            self.show_window()

        def quit_app(icon, item=None):
            icon.stop()
            self.quit_app()

        menu = Menu(
            MenuItem(self._("tray_show"), show_window),
            MenuItem(self._("tray_exit"), quit_app)
        )
        self.tray_icon = Icon("Crypto Tracker", icon=create_image(), title="Crypto Tracker", menu=menu)

    def toggle_always_on_top(self):
        new_state = not self.config["always_on_top"]
        self.config["always_on_top"] = new_state
        self.root.wm_attributes("-topmost", new_state)
        self.always_on_top_btn.config(bg="#333" if new_state else "#444")
        self.save_config(self.config)

    def open_settings(self):
        SettingsWindow(self.root, self.config, self.save_config, self.translations)

    def open_advanced_settings(self):
        AdvancedSettingsWindow(self.root, self.config, self.save_config, self.translations)

    def open_converter(self):
        ConverterWindow(self.root, self.converter, self.monitor, self.translations)

    def show_alert_history(self):
        AlertHistoryWindow(self.root, list(self.alert_history))

    def hide_to_tray(self):
        if HAS_TRAY:
            self.root.withdraw()
            if not hasattr(self, 'tray_thread') or not self.tray_thread.is_alive():
                self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
                self.tray_thread.start()

    def show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def on_close(self):
        if HAS_TRAY:
            self.hide_to_tray()
        else:
            self.quit_app()

    def quit_app(self):
        self.monitor.running = False
        if hasattr(self, 'tray_icon'):
            self.tray_icon.stop()
        self.root.quit()
        self.root.destroy()

    def play_alert_sound(self):
        if self.config.get("show_sound", True):
            if HAS_WINSOUND:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    return
                except:
                    pass
            try:
                print('\a', end='', flush=True)
            except:
                pass

    def show_notification(self, message):
        if self.config.get("show_notifications", True) and HAS_NOTIFICATIONS:
            try:
                notification.notify(
                    title="Crypto Price Alert",
                    message=message,
                    app_name="Crypto Tracker",
                    timeout=5
                )
            except Exception as e:
                logging.error(f"Notification error: {e}")

    def format_price(self, price, currency="USD"):
        if currency == "RUB":
            return f"₽{price:,.0f}"
        elif currency == "EUR":
            return f"€{price:,.2f}"
        else:
            return f"${price:,.2f}"

    def on_chart_resize(self, event):
        self.draw_chart()

    def draw_chart(self):
        if not self.config.get("show_chart", True):
            return
        now = time.time()
        if now - self.last_chart_draw < self.chart_throttle_sec:
            return
        self.last_chart_draw = now

        if not self.chart_history:
            return

        canvas = self.chart_canvas
        canvas.delete("chart")

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        points = list(self.chart_history)
        if len(points) < 2:
            return

        prices = [p[1] for p in points]
        min_p, max_p = min(prices), max(prices)
        if min_p == max_p:
            max_p = min_p * 1.01

        def y_coord(price):
            return height - ((price - min_p) / (max_p - min_p)) * height

        # Сетка
        for i in range(1, 5):
            y = i * height / 5
            canvas.create_line(0, y, width, y, fill="#3a3a3a", tags="chart")

        # График
        x_step = width / (len(points) - 1) if len(points) > 1 else 0
        coords = []
        for i, (_, price) in enumerate(points):
            x = i * x_step
            y = y_coord(price)
            coords.extend([x, y])

        if len(coords) >= 4:
            canvas.create_line(coords, fill="#FFD700", width=2, smooth=True, tags="chart")

        # Метки
        canvas.create_text(5, 5, anchor="nw", text=f"{self.format_price(max_p)}", fill="#666", font=("Helvetica", 8),
                           tags="chart")
        canvas.create_text(5, height - 15, anchor="nw", text=f"{self.format_price(min_p)}", fill="#666",
                           font=("Helvetica", 8), tags="chart")

    def on_price_update(self, raw_price, timestamp):
        self.chart_history.append((timestamp, raw_price))

        if time.time() - self.last_gui_update < 1.0 / MAX_GUI_UPDATES_PER_SEC:
            return
        self.last_gui_update = time.time()

        symbol = self.config["symbol"].lower()
        base_currency = symbol[-3:].upper()
        display_price = raw_price

        self.root.after(0, self.update_gui, display_price, raw_price, base_currency, timestamp)

        should_alert, msg = self.monitor.check_alert(raw_price, timestamp, self.config)
        if should_alert:
            full_msg = self._("alert_message").format(msg=msg)
            self.alert_history.append(f"{datetime.now().strftime('%H:%M:%S')} — {full_msg}")
            self.root.after(0, self.trigger_alert, full_msg)

    def trigger_alert(self, message):
        self.play_alert_sound()
        self.show_notification(message)
        self.status_label.config(text=message, fg="orange")

    def update_gui(self, display_price, raw_price, currency, timestamp):
        color = "white"
        arrow = ""
        if self.config.get("color_price", True) and self.monitor.last_price is not None:
            if raw_price > self.monitor.last_price:
                color, arrow = "#00FF00", " ▲"
            elif raw_price < self.monitor.last_price:
                color, arrow = "#FF3333", " ▼"

        self.price_label.config(text=f"{self.format_price(display_price, currency)}{arrow}", fg=color)
        self.monitor.last_price = raw_price

        if currency == "USD":
            eur = raw_price * self.converter.get_rate("EUR")
            rub = raw_price * self.converter.get_rate("RUB")
            self.converted_label.config(text=f"≈ €{eur:,.2f} | ₽{rub:,.0f}")
        else:
            self.converted_label.config(text="")

        # Обновляем USDT/RUB
        usdt_rub = self.converter.get_usdt_rub()
        self.usdt_rub_label.config(text=f"USDT/RUB ≈ ₽{usdt_rub:,.2f}")

        self.status_indicator.config(bg="#00FF00")
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        self.status_label.config(text=f"{self._('live_updated')} {time_str}", fg="#00FF00")

        self.draw_chart()

    def on_status_update(self, status, color="orange"):
        self.root.after(0, lambda: self.status_label.config(text=status, fg=color))
        self.root.after(0, lambda: self.status_indicator.config(bg=color if color != "#00FF00" else "red"))

    def rest_fallback_loop(self):
        while self.monitor.running:
            try:
                if not self.monitor.price_history or time.time() - self.monitor.price_history[-1][0] > 15:
                    symbol = self.config["symbol"].upper()
                    url = BINANCE_REST_URL + symbol
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        price = float(data['price'])
                        self.monitor.add_price(price)
                        self.on_status_update("REST fallback", "#00AA00")
            except Exception as e:
                logging.error(f"REST fallback error: {e}")
            time.sleep(10)

    def start_websocket(self):
        retry_delay = 5
        while self.monitor.running:
            try:
                def on_message(ws, message):
                    if not self.monitor.running:
                        return
                    try:
                        data = json.loads(message)
                        price = float(data['c'])
                        volume = float(data.get('v', 0))
                        self.monitor.add_price(price, volume)
                    except Exception as e:
                        logging.error(f"Message error: {e}")

                def on_error(ws, error):
                    if self.monitor.running:
                        self.on_status_update(self._("error"), "red")

                def on_close(ws, code, msg):
                    if self.monitor.running:
                        self.on_status_update(self._("disconnected"), "orange")

                def on_open(ws):
                    if self.monitor.running:
                        self.on_status_update(self._("connected"), "#00FF00")
                        retry_delay = 5

                symbol = self.config["symbol"].lower()
                ws_url = f"{BINANCE_WS_URL}{symbol}@ticker"
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
            if self.monitor.running:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 60)


if __name__ == "__main__":
    # Проверка зависимостей для AI
    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        AI_AVAILABLE = True
    except ImportError:
        AI_AVAILABLE = False
        print("ВНИМАНИЕ: AI функции недоступны. Установите: pip install numpy scikit-learn")
        logging.warning("AI dependencies not available")

    root = tk.Tk()
    app = CryptoPriceApp(root)
    root.mainloop()