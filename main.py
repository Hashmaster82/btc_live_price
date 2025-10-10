"""Главный модуль приложения."""
import tkinter as tk
from tkinter import font
import threading
import time
import urllib.parse
import requests
from typing import Dict, Any, List, Optional

from config import ConfigManager
from models import PriceMonitor, CurrencyConverter, AlertConfig, Observable
from services import BinancePriceProvider, NewsService, NotificationService
from utils import Internationalization, FormatUtils, ErrorHandler
from views import AlertHistoryWindow, ConverterWindow, ChartWindow, SettingsWindow

# Проверка дополнительных библиотек
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


class CryptoPriceApp(Observable):
    """Главное приложение крипто-трекера."""

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.alert_history: List[str] = []
        self.news_cache: List[str] = []

        # Инициализация сервисов
        self.config_manager = ConfigManager()
        self.translations = Internationalization()
        self.converter = CurrencyConverter()
        self.notification_service = NotificationService()
        self.news_service = NewsService()
        self.price_provider = BinancePriceProvider()

        # Инициализация монитора цен ДО создания UI
        self.monitor = PriceMonitor(self.config_manager.get("symbol"))
        self.monitor.add_observer(self._on_price_event)

        # Настройка языка
        self._setup_language()

        # Инициализация UI
        self._setup_ui()

        # Запуск сервисов
        self._start_services()

        # Настройка системного трея
        if HAS_TRAY:
            self._setup_tray()

        # Обработка закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._setup_keybindings()

    def _setup_language(self):
        """Настраивает язык приложения."""
        language = self.config_manager.get("language", "auto")
        self.translations.set_language(language)
        self.root.title(self.translations.get_text("title"))

    def _setup_ui(self):
        """Настраивает пользовательский интерфейс."""
        self.root.geometry("520x420")
        self.root.resizable(False, False)
        self._apply_theme()

        if self.config_manager.get("always_on_top", False):
            self.root.wm_attributes("-topmost", True)

        # Создание элементов UI
        self._create_title()
        self._create_price_display()
        self._create_portfolio_display()
        self._create_currency_info()
        self._create_news_section()
        self._create_status_bar()
        self._create_buttons()

        self._set_icon()

    def _create_title(self):
        """Создает заголовок с торговой парой."""
        self.title_label = tk.Label(
            self.root, text="", font=font.Font(size=18, weight="bold"),
            bg="#1e1e1e", fg="#FFD700"
        )
        self._update_title()
        self.title_label.pack(pady=8)

    def _create_price_display(self):
        """Создает отображение цены."""
        self.price_label = tk.Label(
            self.root, text="Loading...",
            font=font.Font(family="Consolas", size=24, weight="bold"),
            bg="#1e1e1e", fg="white"
        )
        self.price_label.pack()

        self.converted_label = tk.Label(
            self.root, text="", font=("Consolas", 14),
            bg="#1e1e1e", fg="#aaa"
        )
        self.converted_label.pack(pady=(0, 5))

    def _create_portfolio_display(self):
        """Создает отображение портфеля."""
        self.portfolio_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.portfolio_frame.pack(pady=5)

        self.portfolio_label = tk.Label(
            self.portfolio_frame, text="",
            font=("Consolas", 14, "bold"),
            bg="#1e1e1e", fg="#00FF00"
        )
        self.portfolio_label.pack()
        # Не вызываем _update_portfolio_display здесь - будет вызвано при обновлении цены
        self._update_portfolio_display()

    def _create_currency_info(self):
        """Создает отображение информации о валютах."""
        usdt_rub = self.converter.get_usdt_rub()
        self.usdt_rub_label = tk.Label(
            self.root, text=f"USDT/RUB ≈ ₽{usdt_rub:,.2f}",
            font=("Consolas", 10), bg="#1e1e1e", fg="#888"
        )
        self.usdt_rub_label.pack(pady=(2, 0))

    def _create_news_section(self):
        """Создает секцию новостей."""
        news_frame = tk.Frame(self.root, bg="#1e1e1e")
        news_frame.pack(pady=(10, 0), padx=20, fill="x")

        tk.Label(news_frame, text="📰 Последние новости",
                 bg="#1e1e1e", fg="#FFD700",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.news_label = tk.Label(news_frame, text="Загрузка новостей...", bg="#1e1e1e",
                                   fg="#ccc", font=("Arial", 9),
                                   justify="left", wraplength=480)
        self.news_label.pack(anchor="w", pady=(5, 0))

    def _create_status_bar(self):
        """Создает статус бар."""
        status_frame = tk.Frame(self.root, bg="#1e1e1e")
        status_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        self.status_indicator = tk.Label(status_frame, width=2, bg="orange")
        self.status_indicator.pack(side="left", padx=(0, 5))

        self.status_label = tk.Label(status_frame,
                                     text=self.translations.get_text("connecting"),
                                     fg="#888", bg="#1e1e1e")
        self.status_label.pack(side="left")

    def _create_buttons(self):
        """Создает кнопки управления."""
        btn_container = tk.Frame(self.root, bg="#1e1e1e")
        btn_container.pack(side="bottom", pady=8)

        # Первый ряд кнопок
        btn_row1 = tk.Frame(btn_container, bg="#1e1e1e")
        btn_row1.pack(pady=2)

        if HAS_TRAY:
            btn_min = tk.Button(btn_row1, text=self.translations.get_text("btn_minimize"),
                                command=self.hide_to_tray, bg="#333", fg="white",
                                relief="flat", width=14)
            btn_min.pack(side="left", padx=2)

        self.always_on_top_btn = tk.Button(
            btn_row1, text=self.translations.get_text("btn_always_on_top"),
            command=self.toggle_always_on_top,
            bg="#333" if self.config_manager.get("always_on_top") else "#444",
            fg="white", relief="flat", width=14
        )
        self.always_on_top_btn.pack(side="left", padx=2)

        btn_settings = tk.Button(btn_row1, text=self.translations.get_text("btn_settings"),
                                 command=self.open_settings, bg="#333", fg="white",
                                 relief="flat", width=14)
        btn_settings.pack(side="left", padx=2)

        # Второй ряд кнопок
        btn_row2 = tk.Frame(btn_container, bg="#1e1e1e")
        btn_row2.pack(pady=2)

        btn_history = tk.Button(btn_row2, text="История",
                                command=self.show_alert_history,
                                bg="#333", fg="white", relief="flat", width=14)
        btn_history.pack(side="left", padx=2)

        btn_converter = tk.Button(btn_row2, text="Конвертер",
                                  command=self.open_converter,
                                  bg="#333", fg="white", relief="flat", width=14)
        btn_converter.pack(side="left", padx=2)

        btn_chart = tk.Button(btn_row2, text="График",
                              command=self.open_chart,
                              bg="#333", fg="white", relief="flat", width=14)
        btn_chart.pack(side="left", padx=2)

    def _apply_theme(self):
        """Применяет тему оформления."""
        self.root.configure(bg="#1e1e1e")

    def _set_icon(self):
        """Устанавливает иконку приложения."""
        try:
            from PIL import Image, ImageDraw, ImageTk
            img = Image.new("RGBA", (16, 16), (30, 30, 30, 255))
            draw = ImageDraw.Draw(img)
            draw.rectangle([4, 4, 11, 11], outline=(255, 215, 0), width=1)
            draw.text((5, 2), "C", fill=(255, 215, 0))
            icon = ImageTk.PhotoImage(img)
            self.root.iconphoto(True, icon)
        except Exception as e:
            ErrorHandler.handle_ui_error(e, "icon")

    def _start_services(self):
        """Запускает фоновые сервисы."""
        # Запуск WebSocket
        symbol = self.config_manager.get("symbol")
        self.price_provider.start_stream(symbol, self._on_price_update)

        # Запуск обновления новостей
        self.news_thread = threading.Thread(target=self._news_update_loop, daemon=True)
        self.news_thread.start()

        # Запуск фолбэка REST
        self.rest_thread = threading.Thread(target=self._rest_fallback_loop, daemon=True)
        self.rest_thread.start()

    def _on_price_event(self, event: str, data: Dict):
        """Обрабатывает события от монитора цен."""
        if event == 'price_update':
            self.root.after(0, self._process_price_update, data)

    def _process_price_update(self, data: Dict):
        """Обрабатывает обновление цены в UI потоке."""
        price = data['price']
        timestamp = data['timestamp']

        # Обновление UI
        self._update_price_display(price, timestamp)

        # Проверка оповещений
        alert_config = AlertConfig(
            threshold_up=self.config_manager.get("alert_threshold_up", 1.0),
            threshold_down=self.config_manager.get("alert_threshold_down", 1.0),
            absolute_up=self.config_manager.get("alert_absolute_up"),
            absolute_down=self.config_manager.get("alert_absolute_down"),
            cooldown_sec=self.config_manager.get("alert_cooldown_sec", 15)
        )

        should_alert, message = self.monitor.check_alert(price, timestamp, alert_config)
        if should_alert:
            self._trigger_alert(message)

    def _trigger_alert(self, message: str):
        """Активирует оповещение."""
        full_message = self.translations.get_text("alert_message", msg=message)

        # Добавление в историю
        self.alert_history.append(f"{time.strftime('%H:%M:%S')} — {full_message}")

        # Воспроизведение звука
        if self.config_manager.get("show_sound", True):
            self.notification_service.play_alert()

        # Системное уведомление
        if self.config_manager.get("show_notifications", True):
            self.notification_service.show_notification("Crypto Price Alert", full_message)

        # Обновление статуса
        self.status_label.config(text=full_message, fg="orange")

    def _update_price_display(self, price: float, timestamp: float):
        """Обновляет отображение цены."""
        symbol = self.config_manager.get("symbol").lower()
        currency = symbol[-3:].upper()

        # Цвет и стрелка направления
        color = "white"
        arrow = ""
        if self.config_manager.get("color_price", True) and self.monitor.last_price is not None:
            if price > self.monitor.last_price:
                color, arrow = "#00FF00", " ▲"
            elif price < self.monitor.last_price:
                color, arrow = "#FF3333", " ▼"

        self.price_label.config(
            text=f"{FormatUtils.format_price(price, currency)}{arrow}",
            fg=color
        )

        # Конвертированные цены
        if currency == "USD":
            eur_price = price * 0.92  # Упрощенная конвертация
            rub_price = price * self.converter.get_usdt_rub()
            self.converted_label.config(text=f"≈ €{eur_price:,.2f} | ₽{rub_price:,.0f}")

        # Обновление курса USDT/RUB
        usdt_rub = self.converter.get_usdt_rub()
        self.usdt_rub_label.config(text=f"USDT/RUB ≈ ₽{usdt_rub:,.2f}")

        # Обновление портфеля
        self._update_portfolio_display()

        # Обновление статуса
        self.status_indicator.config(bg="#00FF00")
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        self.status_label.config(
            text=f"{self.translations.get_text('live_updated')} {time_str}",
            fg="#00FF00"
        )

    def _update_portfolio_display(self):
        """Обновляет отображение портфеля."""
        try:
            asset = self.config_manager.get("portfolio_asset", "BTC")
            amount = self.config_manager.get("portfolio_amount", 0.0)

            if amount <= 0:
                self.portfolio_label.config(text="")
                return

            # Проверяем, что монитор инициализирован
            if not hasattr(self, 'monitor') or self.monitor.last_price is None:
                self.portfolio_label.config(text="Ожидание данных...")
                return

            btc_price = self.monitor.last_price or 0
            usdt_rub = self.converter.get_usdt_rub()

            display_text = FormatUtils.format_portfolio_value(amount, asset, btc_price, usdt_rub)
            self.portfolio_label.config(text=display_text)

        except Exception as e:
            ErrorHandler.handle_ui_error(e, "portfolio_display")
            self.portfolio_label.config(text="Ошибка расчета")

    def _update_title(self):
        """Обновляет заголовок с торговой парой."""
        symbol = self.config_manager.get("symbol").upper()
        if symbol.endswith('USDT'):
            display_symbol = f"{symbol[:-4]}/USDT"
        elif symbol.endswith('EUR'):
            display_symbol = f"{symbol[:-3]}/EUR"
        elif symbol.endswith('RUB'):
            display_symbol = f"{symbol[:-3]}/RUB"
        else:
            display_symbol = symbol

        self.title_label.config(text=display_symbol)

    def _on_price_update(self, price: Optional[float], timestamp: float, **kwargs):
        """Callback для обновлений цены."""
        if price is not None:
            self.monitor.add_price(price)

        # Обработка статусов соединения
        if kwargs.get('connected'):
            self.status_label.config(text=self.translations.get_text("connected"), fg="#00FF00")
        elif kwargs.get('disconnected'):
            self.status_label.config(text=self.translations.get_text("disconnected"), fg="orange")
        elif kwargs.get('error'):
            self.status_label.config(text=self.translations.get_text("error"), fg="red")

    def _news_update_loop(self):
        """Цикл обновления новостей."""
        while getattr(self, 'monitor', None) and self.monitor.running:
            try:
                language = "ru" if self.config_manager.get("language") == "ru" else "en"
                news = self.news_service.fetch_news(language=language)

                if news:
                    self.news_cache = news
                    self.root.after(0, self._update_news_display)

                time.sleep(300)  # 5 минут
            except Exception as e:
                ErrorHandler.handle_ui_error(e, "news")
                time.sleep(60)

    def _update_news_display(self):
        """Обновляет отображение новостей."""
        if hasattr(self, 'news_cache') and self.news_cache:
            news_text = "\n".join(f"• {item}" for item in self.news_cache[:3])
            self.news_label.config(text=news_text)
        else:
            self.news_label.config(text="Нет новостей")

    def _rest_fallback_loop(self):
        """Цикл фолбэка через REST API."""
        while getattr(self, 'monitor', None) and self.monitor.running:
            try:
                # Проверяем, есть ли свежие данные
                if (not hasattr(self.monitor, 'price_history') or
                        not self.monitor.price_history or
                        time.time() - self.monitor.price_history[-1][0] > 15):

                    symbol = self.config_manager.get("symbol").upper()
                    current_price = self.price_provider.get_current_price(symbol)

                    if current_price:
                        self.monitor.add_price(current_price)
                        self.status_label.config(text="REST fallback", fg="#00AA00")

                time.sleep(10)
            except Exception as e:
                ErrorHandler.handle_ui_error(e, "rest_fallback")
                time.sleep(30)

    def _setup_keybindings(self):
        """Настраивает горячие клавиши."""
        self.root.bind('<Control-q>', lambda e: self.quit_app())
        self.root.bind('<Control-h>', lambda e: self.show_window())
        self.root.bind('<Control-l>', lambda e: self.show_alert_history())

    def _setup_tray(self):
        """Настраивает системный трей."""
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
            MenuItem(self.translations.get_text("tray_show"), show_window),
            MenuItem(self.translations.get_text("tray_exit"), quit_app)
        )

        self.tray_icon = Icon("Crypto Tracker", icon=create_image(),
                              title="Crypto Tracker", menu=menu)

    # Public methods
    def toggle_always_on_top(self):
        """Переключает режим 'Поверх всех окон'."""
        new_state = not self.config_manager.get("always_on_top", False)
        self.config_manager.set("always_on_top", new_state)
        self.root.wm_attributes("-topmost", new_state)
        self.always_on_top_btn.config(bg="#333" if new_state else "#444")

    def open_settings(self):
        """Открывает окно настроек."""
        SettingsWindow(self.root, self.config_manager.all_config,
                       self._on_config_save, self.translations)

    def open_converter(self):
        """Открывает окно конвертера."""
        ConverterWindow(self.root, self.converter, self.monitor, self.translations)

    def open_chart(self):
        """Открывает окно графика."""
        ChartWindow(self.root, self.config_manager.get("symbol"), self.translations)

    def show_alert_history(self):
        """Показывает историю оповещений."""
        AlertHistoryWindow(self.root, self.alert_history, self.translations)

    def hide_to_tray(self):
        """Скрывает окно в системный трей."""
        if HAS_TRAY:
            self.root.withdraw()
            if not hasattr(self, 'tray_thread') or not self.tray_thread.is_alive():
                self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
                self.tray_thread.start()

    def show_window(self):
        """Показывает окно из трея."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def on_close(self):
        """Обрабатывает закрытие окна."""
        if HAS_TRAY:
            self.hide_to_tray()
        else:
            self.quit_app()

    def quit_app(self):
        """Завершает работу приложения."""
        if hasattr(self, 'monitor'):
            self.monitor.running = False

        if hasattr(self, 'price_provider'):
            self.price_provider.stop_stream()

        if hasattr(self, 'tray_icon'):
            self.tray_icon.stop()

        self.root.quit()
        self.root.destroy()

    def _on_config_save(self, new_config: Dict[str, Any]):
        """Обрабатывает сохранение конфигурации."""
        self.config_manager.update(new_config)

        # Обновление языка при необходимости
        if 'language' in new_config:
            self.translations.set_language(new_config['language'])
            self.root.title(self.translations.get_text("title"))

        # Обновление заголовка при изменении символа
        if 'symbol' in new_config:
            self._update_title()
            # Перезапуск WebSocket с новым символом
            self.price_provider.stop_stream()
            self.price_provider.start_stream(new_config['symbol'], self._on_price_update)

        # Обновление портфеля
        self._update_portfolio_display()


def main():
    """Точка входа в приложение."""
    root = tk.Tk()
    app = CryptoPriceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()