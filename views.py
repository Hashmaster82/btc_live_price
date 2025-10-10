"""Классы пользовательского интерфейса."""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List, Optional, Callable
import threading
import urllib.parse
import requests
import time  # Добавить этот импорт

from config import POPULAR_SYMBOLS  # Исправить импорт
from models import PriceMonitor, CurrencyConverter, AlertConfig
from services import NotificationService, BinancePriceProvider, NewsService
from utils import Internationalization, FormatUtils, ValidationUtils, ErrorHandler

# Попытка импорта дополнительных библиотек
try:
    from pystray import Icon, Menu, MenuItem
    from PIL import Image, ImageDraw

    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

try:
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import mplfinance as mpf
    import pandas as pd

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class BaseWindow:
    """Базовый класс для окон."""

    def __init__(self, parent, translations: Internationalization):
        self.parent = parent
        self.translations = translations
        self._ = translations.get_text
        self.window = None

    def show(self):
        """Показывает окно."""
        if self.window:
            self.window.deiconify()
            self.window.lift()

    def hide(self):
        """Скрывает окно."""
        if self.window:
            self.window.withdraw()

    def close(self):
        """Закрывает окно."""
        if self.window:
            self.window.destroy()


class AlertHistoryWindow(BaseWindow):
    """Окно истории оповещений."""

    def __init__(self, parent, alerts: List[str], translations: Internationalization):
        super().__init__(parent, translations)
        self.alerts = alerts
        self._create_window()

    def _create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(self._("history_title"))
        self.window.geometry("520x320")
        self.window.configure(bg="#2b2b2b")

        frame = tk.Frame(self.window, bg="#2b2b2b")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            bg="#333",
            fg="white",
            font=("Consolas", 10),
            width=80
        )
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        for alert in reversed(self.alerts):
            self.listbox.insert("end", alert)


class ConverterWindow(BaseWindow):
    """Окно конвертера валют."""

    def __init__(self, parent, converter: CurrencyConverter, price_monitor: PriceMonitor,
                 translations: Internationalization):
        super().__init__(parent, translations)
        self.converter = converter
        self.price_monitor = price_monitor
        self._create_window()

    def _create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(self._("converter_title"))
        self.window.geometry("320x200")
        self.window.configure(bg="#2b2b2b")

        main_frame = tk.Frame(self.window, bg="#2b2b2b")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Ввод amount
        tk.Label(main_frame, text=self._("enter_amount"), bg="#2b2b2b",
                 fg="white", font=("Arial", 10)).pack(anchor="w", pady=(0, 8))

        input_frame = tk.Frame(main_frame, bg="#2b2b2b")
        input_frame.pack(fill="x", pady=5)

        self.amount_var = tk.StringVar()
        self.amount_var.trace("w", self.convert)
        tk.Entry(input_frame, textvariable=self.amount_var, width=20,
                 font=("Arial", 11)).pack(side="left", padx=(0, 10))

        # Режимы конвертации
        mode_frame = tk.Frame(main_frame, bg="#2b2b2b")
        mode_frame.pack(fill="x", pady=8)

        self.mode_var = tk.StringVar(value="btc")
        tk.Radiobutton(mode_frame, text="BTC → RUB", variable=self.mode_var,
                       value="btc", bg="#2b2b2b", fg="white", selectcolor="#444",
                       font=("Arial", 9), command=self.convert).pack(side="left", padx=(0, 15))
        tk.Radiobutton(mode_frame, text="USDT → RUB", variable=self.mode_var,
                       value="usdt", bg="#2b2b2b", fg="white", selectcolor="#444",
                       font=("Arial", 9), command=self.convert).pack(side="left")

        # Результат
        result_frame = tk.Frame(main_frame, bg="#2b2b2b")
        result_frame.pack(fill="x", pady=15)

        tk.Label(result_frame, text="Результат:", bg="#2b2b2b",
                 fg="white", font=("Arial", 10)).pack(anchor="w")

        self.result_label = tk.Label(result_frame, text="", bg="#2b2b2b",
                                     fg="lightgreen", font=("Consolas", 12, "bold"))
        self.result_label.pack(anchor="w", pady=(5, 0))

        # Информация о курсах
        info_frame = tk.Frame(main_frame, bg="#2b2b2b")
        info_frame.pack(fill="x", pady=(10, 0))

        self.rate_info = tk.Label(info_frame, text="", bg="#2b2b2b",
                                  fg="#888", font=("Arial", 8))
        self.rate_info.pack(anchor="w")

        self.update_rate_info()

    def update_rate_info(self):
        """Обновляет информацию о курсах."""
        btc_price = self.price_monitor.last_price or 0
        usdt_rub = self.converter.get_usdt_rub()

        if btc_price > 0:
            self.rate_info.config(text=f"BTC: ${btc_price:,.2f} | USDT/RUB: {usdt_rub:.2f}")
        else:
            self.rate_info.config(text=f"USDT/RUB: {usdt_rub:.2f}")

    def convert(self, *args):
        """Выполняет конвертацию."""
        try:
            amount_text = self.amount_var.get().strip()
            if not amount_text:
                self.result_label.config(text="")
                return

            if not ValidationUtils.validate_amount(amount_text):
                self.result_label.config(text="Введите число")
                return

            amount = float(amount_text)
            mode = self.mode_var.get()
            self.update_rate_info()

            if mode == "btc":
                btc_price_usd = self.price_monitor.last_price or 0
                if btc_price_usd <= 0:
                    self.result_label.config(text="Нет данных о цене BTC")
                    return

                usdt_rub_rate = self.converter.get_usdt_rub()
                rub_amount = amount * btc_price_usd * usdt_rub_rate
                self.result_label.config(text=f"≈ {rub_amount:,.0f} RUB")
            else:
                usdt_rub_rate = self.converter.get_usdt_rub()
                rub_amount = amount * usdt_rub_rate
                self.result_label.config(text=f"≈ {rub_amount:,.0f} RUB")

        except Exception as e:
            ErrorHandler.handle_ui_error(e, "converter")
            self.result_label.config(text="Ошибка конвертации")


class ChartWindow(BaseWindow):
    """Окно графиков."""

    def __init__(self, parent, symbol: str, translations: Internationalization):
        super().__init__(parent, translations)
        self.symbol = symbol
        self._create_window()

    def _create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"{self._('chart_title')} - {self.symbol.upper()}")
        self.window.state('zoomed')
        self.window.configure(bg="#1e1e1e")

        # Фрейм для графика
        chart_frame = tk.Frame(self.window, bg="#1e1e1e")
        chart_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        # Фрейм для анализа
        analysis_frame = tk.Frame(self.window, bg="#1e1e1e")
        analysis_frame.pack(fill="x", padx=10, pady=10)

        # Загрузка данных
        self.load_and_display_chart(chart_frame, analysis_frame)

        # Кнопка закрытия
        close_btn = tk.Button(self.window, text="Закрыть", command=self.close,
                              bg="#333", fg="white", relief="flat")
        close_btn.pack(pady=5)

    def load_and_display_chart(self, chart_frame, analysis_frame):
        """Загружает и отображает график."""
        if not HAS_MPL:
            self._show_error(chart_frame, "Библиотеки для графиков не установлены")
            return

        try:
            # Получение данных
            data = self._fetch_chart_data()
            if not data:
                raise Exception("Не удалось загрузить данные")

            # Построение графика
            df = self._create_chart(chart_frame, data)
            # Анализ
            self._generate_analysis(analysis_frame, df)

        except Exception as e:
            ErrorHandler.handle_ui_error(e, "chart")
            self._show_error(chart_frame, f"Ошибка загрузки данных: {str(e)}")

    def _fetch_chart_data(self):
        """Получает данные для графика."""
        end_time = int(time.time() * 1000)
        start_time = end_time - (180 * 24 * 60 * 60 * 1000)  # 180 дней

        params = {
            'symbol': self.symbol.upper(),
            'interval': '1d',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 200
        }

        url = f"https://api.binance.com/api/v3/klines?{urllib.parse.urlencode(params)}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")

        return response.json()

    def _create_chart(self, parent, data):
        """Создает свечной график."""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Конвертируем в float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Настройка стиля
        mc = mpf.make_marketcolors(
            up='#00FF00', down='#FF3333',
            edge={'up': '#00CC00', 'down': '#CC0000'},
            wick={'up': '#00CC00', 'down': '#CC0000'},
            volume='in'
        )

        s = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mc,
            facecolor='#1e1e1e',
            figcolor='#1e1e1e',
            gridcolor='#333333',
            gridstyle='-'
        )

        # Создание графика
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=s,
            title=f'{self.symbol.upper()} - Последние 6 месяцев',
            ylabel='Цена (USD)',
            volume=True,
            ylabel_lower='Объем',
            returnfig=True,
            datetime_format='%b %Y',
            xrotation=0
        )

        # Настройка цветов
        for ax in axes:
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            if hasattr(ax, 'title'):
                ax.title.set_color('white')

        # Встраивание в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        return df

    def _generate_analysis(self, parent, df):
        """Генерирует технический анализ."""
        try:
            # Простой технический анализ
            current_price = df['close'].iloc[-1]

            # Скользящие средние
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1]
            ma_200 = df['close'].rolling(window=200).mean().iloc[-1]

            # RSI
            rsi = self._calculate_rsi(df['close'])

            # Волатильность
            volatility = df['close'].pct_change().std() * (252 ** 0.5)

            # Определение тренда
            if current_price > ma_50 > ma_200:
                trend = "бычий"
                trend_color = "#00FF00"
            elif current_price < ma_50 < ma_200:
                trend = "медвежий"
                trend_color = "#FF3333"
            else:
                trend = "боковой"
                trend_color = "#FFFF00"

            # Простое предсказание
            if current_price > ma_50:
                prediction = "рост"
                pred_color = "#00FF00"
            else:
                prediction = "падение"
                pred_color = "#FF3333"

            # Текст анализа
            analysis_text = f"""
Профессиональный анализ ({df.index[-1].strftime('%d.%m.%Y')}):
• Текущая цена: ${current_price:,.2f}
• Тренд: {trend} (MA50: ${ma_50:,.2f}, MA200: ${ma_200:,.2f})
• RSI: {rsi:.2f} ({"перекупленность" if rsi > 70 else "перепроданность" if rsi < 30 else "нейтрально"})
• Волатильность: {volatility:.2%} годовых

AI-предсказание:
На основе технических индикаторов ожидается {prediction} цены в краткосрочной перспективе.
Рекомендуется {"покупать" if prediction == "рост" else "продавать или ожидать"}.
            """

            analysis_label = tk.Label(
                parent,
                text=analysis_text.strip(),
                bg="#1e1e1e",
                fg="white",
                font=("Consolas", 10),
                justify="left",
                wraplength=1200
            )
            analysis_label.pack(pady=10)

            # Цветные индикаторы
            trend_frame = tk.Frame(parent, bg="#1e1e1e")
            trend_frame.pack()
            tk.Label(trend_frame, text="Тренд:", bg="#1e1e1e", fg="white").pack(side="left")
            tk.Label(trend_frame, text=trend, bg="#1e1e1e", fg=trend_color,
                     font=("Arial", 10, "bold")).pack(side="left", padx=5)

            pred_frame = tk.Frame(parent, bg="#1e1e1e")
            pred_frame.pack()
            tk.Label(pred_frame, text="Предсказание:", bg="#1e1e1e", fg="white").pack(side="left")
            tk.Label(pred_frame, text=prediction, bg="#1e1e1e", fg=pred_color,
                     font=("Arial", 10, "bold")).pack(side="left", padx=5)

        except Exception as e:
            ErrorHandler.handle_ui_error(e, "chart_analysis")

    def _calculate_rsi(self, prices, window=14):
        """Рассчитывает RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50

    def _show_error(self, parent, message):
        """Показывает сообщение об ошибке."""
        error_label = tk.Label(
            parent,
            text=message,
            bg="#1e1e1e",
            fg="red",
            font=("Arial", 12)
        )
        error_label.pack(pady=20)


class SettingsWindow(BaseWindow):
    """Окно настроек."""

    def __init__(self, parent, config: Dict[str, Any], save_callback: Callable,
                 translations: Internationalization):
        super().__init__(parent, translations)
        self.config = config.copy()
        self.save_callback = save_callback
        self._create_window()

    def _create_window(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title(self._("settings_title"))
        self.window.geometry("440x580")
        self.window.resizable(False, False)
        self.window.configure(bg="#2b2b2b")
        self.window.transient(self.parent)
        self.window.grab_set()

        # Создаем скроллируемую область
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

        # Настройка скролла мышью
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.window.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Содержимое настроек
        self._create_settings_content(scrollable_frame)

    def _create_settings_content(self, parent):
        """Создает содержимое окна настроек."""

        # Язык
        tk.Label(parent, text=self._("language"), bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        self.lang_var = tk.StringVar(value=self.config.get("language", "auto"))
        lang_frame = tk.Frame(parent, bg="#2b2b2b")
        lang_frame.pack(fill="x", padx=20, pady=(0, 10))

        tk.Radiobutton(lang_frame, text="Авто", variable=self.lang_var, value="auto",
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Radiobutton(lang_frame, text="Русский", variable=self.lang_var, value="ru",
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")
        tk.Radiobutton(lang_frame, text="English", variable=self.lang_var, value="en",
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Торговая пара
        tk.Label(parent, text=self._("trading_pair"), bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        self.symbol_var = tk.StringVar(value=self.config.get("symbol", "btcusdt"))
        symbol_combo = ttk.Combobox(parent, textvariable=self.symbol_var,
                                    values=POPULAR_SYMBOLS, state="readonly", width=20)
        symbol_combo.pack(padx=20, pady=(0, 10))

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Портфель
        tk.Label(parent, text="Портфель", bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        portfolio_frame = tk.Frame(parent, bg="#2b2b2b")
        portfolio_frame.pack(fill="x", padx=20, pady=(0, 10))

        # Актив
        asset_frame = tk.Frame(portfolio_frame, bg="#2b2b2b")
        asset_frame.pack(fill="x", pady=(0, 5))

        self.asset_var = tk.StringVar(value=self.config.get("portfolio_asset", "BTC"))
        tk.Radiobutton(asset_frame, text="BTC", variable=self.asset_var, value="BTC",
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(side="left", padx=(0, 15))
        tk.Radiobutton(asset_frame, text="USDT", variable=self.asset_var, value="USDT",
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(side="left")

        # Количество
        amount_frame = tk.Frame(portfolio_frame, bg="#2b2b2b")
        amount_frame.pack(fill="x", pady=5)

        tk.Label(amount_frame, text="Количество:", bg="#2b2b2b", fg="white").pack(side="left")
        self.asset_amount_var = tk.StringVar(value=str(self.config.get("portfolio_amount", "0")))
        tk.Entry(amount_frame, textvariable=self.asset_amount_var, width=12).pack(side="left", padx=(5, 0))

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Относительные пороги
        tk.Label(parent, text="Относительные оповещения", bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        # Порог вверх
        tk.Label(parent, text=self._("alert_threshold_up"), bg="#2b2b2b", fg="white").pack(pady=(5, 0))

        self.up_var = tk.DoubleVar(value=self.config.get("alert_threshold_up", 1.0))
        up_scale = ttk.Scale(parent, from_=0.1, to=10.0, variable=self.up_var,
                             orient="horizontal")
        up_scale.pack(fill="x", padx=20, pady=5)

        self.up_label = tk.Label(parent, text=f"{self.up_var.get():.1f}%",
                                 bg="#2b2b2b", fg="#aaa")
        self.up_label.pack()
        self.up_var.trace("w", lambda *a: self.up_label.config(text=f"{self.up_var.get():.1f}%"))

        # Порог вниз
        tk.Label(parent, text=self._("alert_threshold_down"), bg="#2b2b2b", fg="white").pack(pady=(10, 0))

        self.down_var = tk.DoubleVar(value=self.config.get("alert_threshold_down", 1.0))
        down_scale = ttk.Scale(parent, from_=0.1, to=10.0, variable=self.down_var,
                               orient="horizontal")
        down_scale.pack(fill="x", padx=20, pady=5)

        self.down_label = tk.Label(parent, text=f"{self.down_var.get():.1f}%",
                                   bg="#2b2b2b", fg="#aaa")
        self.down_label.pack()
        self.down_var.trace("w", lambda *a: self.down_label.config(text=f"{self.down_var.get():.1f}%"))

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Абсолютные пороги
        tk.Label(parent, text="Абсолютные оповещения", bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        # Абсолютный верхний порог
        tk.Label(parent, text=self._("alert_absolute_up"), bg="#2b2b2b", fg="white").pack(pady=(5, 0))

        abs_up_val = self.config.get("alert_absolute_up")
        self.abs_up_var = tk.StringVar(value=str(abs_up_val) if abs_up_val is not None else "")
        tk.Entry(parent, textvariable=self.abs_up_var, width=15).pack(pady=5)

        # Абсолютный нижний порог
        tk.Label(parent, text=self._("alert_absolute_down"), bg="#2b2b2b", fg="white").pack(pady=(5, 0))

        abs_down_val = self.config.get("alert_absolute_down")
        self.abs_down_var = tk.StringVar(value=str(abs_down_val) if abs_down_val is not None else "")
        tk.Entry(parent, textvariable=self.abs_down_var, width=15).pack(pady=5)

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Уведомления
        tk.Label(parent, text="Уведомления", bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        notify_frame = tk.Frame(parent, bg="#2b2b2b")
        notify_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.sound_var = tk.BooleanVar(value=self.config.get("show_sound", True))
        tk.Checkbutton(notify_frame, text=self._("sound_enabled"), variable=self.sound_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")

        self.notify_var = tk.BooleanVar(value=self.config.get("show_notifications", True))
        tk.Checkbutton(notify_frame, text=self._("notifications_enabled"), variable=self.notify_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")

        # Разделитель
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Отображение
        tk.Label(parent, text=self._("display_options"), bg="#2b2b2b", fg="white",
                 font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w")

        display_frame = tk.Frame(parent, bg="#2b2b2b")
        display_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.color_price_var = tk.BooleanVar(value=self.config.get("color_price", True))
        tk.Checkbutton(display_frame, text=self._("color_price"), variable=self.color_price_var,
                       bg="#2b2b2b", fg="white", selectcolor="#444").pack(anchor="w")

        # Кнопка сохранения
        tk.Button(parent, text=self._("save"), command=self.save,
                  bg="#444", fg="white", relief="flat", width=20,
                  font=("Arial", 10, "bold")).pack(pady=20)

    def save(self):
        """Сохраняет настройки."""
        try:
            # Валидация абсолютных порогов
            abs_up_str = self.abs_up_var.get().strip()
            abs_down_str = self.abs_down_var.get().strip()

            abs_up = float(abs_up_str) if abs_up_str else None
            abs_down = float(abs_down_str) if abs_down_str else None

            # Валидация количества
            amount_str = self.asset_amount_var.get().strip()
            amount = float(amount_str) if amount_str else 0.0

            if amount < 0:
                messagebox.showerror("Ошибка", "Количество не может быть отрицательным")
                return

            # Создаем новую конфигурацию
            new_config = {
                "language": self.lang_var.get(),
                "symbol": self.symbol_var.get().lower(),
                "portfolio_asset": self.asset_var.get(),
                "portfolio_amount": amount,
                "alert_threshold_up": self.up_var.get(),
                "alert_threshold_down": self.down_var.get(),
                "alert_absolute_up": abs_up,
                "alert_absolute_down": abs_down,
                "show_sound": self.sound_var.get(),
                "show_notifications": self.notify_var.get(),
                "color_price": self.color_price_var.get()
            }

            self.save_callback(new_config)
            messagebox.showinfo("Успех", "Настройки сохранены")
            self.close()

        except ValueError as e:
            messagebox.showerror("Ошибка", "Проверьте правильность введенных числовых значений")
        except Exception as e:
            ErrorHandler.handle_ui_error(e, "settings_save")
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")