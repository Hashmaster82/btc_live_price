"""Модульные тесты для сервисов."""
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import websocket

from services import BinancePriceProvider, NewsService, NotificationService


class TestBinancePriceProvider:
    """Тесты для поставщика цен Binance."""

    def test_initialization(self):
        """Тест инициализации поставщика."""
        provider = BinancePriceProvider()

        assert provider.ws is None
        assert provider.callback is None
        assert provider.running is False
        assert provider.thread is None

    @patch('services.requests.get')
    def test_get_current_price_success(self, mock_get):
        """Тест успешного получения текущей цены."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'price': '50123.45'}
        mock_get.return_value = mock_response

        provider = BinancePriceProvider()
        price = provider.get_current_price("btcusdt")

        assert price == 50123.45
        mock_get.assert_called_once_with(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=5
        )

    @patch('services.requests.get')
    def test_get_current_price_failure(self, mock_get):
        """Тест неудачного получения цены."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        provider = BinancePriceProvider()
        price = provider.get_current_price("btcusdt")

        assert price is None

    @patch('services.requests.get')
    def test_get_current_price_exception(self, mock_get):
        """Тест исключения при получении цены."""
        mock_get.side_effect = Exception("Network error")

        provider = BinancePriceProvider()
        price = provider.get_current_price("btcusdt")

        assert price is None

    def test_start_stream(self):
        """Тест запуска потока данных."""
        provider = BinancePriceProvider()
        mock_callback = Mock()

        with patch.object(provider, '_run_websocket'):
            result = provider.start_stream("btcusdt", mock_callback)

            assert result is True
            assert provider.callback == mock_callback
            assert provider.running is True
            assert provider.thread is not None
            assert provider.thread.daemon is True

    def test_stop_stream(self):
        """Тест остановки потока данных."""
        provider = BinancePriceProvider()
        provider.running = True
        provider.thread = Mock()
        provider.ws = Mock()

        provider.stop_stream()

        assert provider.running is False
        provider.ws.close.assert_called_once()
        provider.thread.join.assert_called_once_with(timeout=5)

    @patch('services.websocket.WebSocketApp')
    def test_websocket_message_handling(self, mock_ws_app):
        """Тест обработки сообщений WebSocket."""
        provider = BinancePriceProvider()
        mock_callback = Mock()
        provider.callback = mock_callback
        provider.running = True

        # Создаем mock WebSocket
        mock_ws = MagicMock()
        mock_ws_app.return_value = mock_ws

        # Запускаем поток (будет остановлен сразу)
        provider.running = False
        provider._run_websocket("btcusdt")

        # Проверяем создание WebSocket
        mock_ws_app.assert_called_once()

    def test_websocket_message_processing(self):
        """Тест обработки конкретного сообщения WebSocket."""
        provider = BinancePriceProvider()
        mock_callback = Mock()
        provider.callback = mock_callback
        provider.running = True

        # Тестовое сообщение от Binance
        test_message = json.dumps({
            'c': '50123.45',  # Current price
            'h': '50200.00',  # High price
            'l': '50000.00',  # Low price
            'v': '1000.0'  # Volume
        })

        # Вызываем обработчик сообщения
        provider._run_websocket = Mock()
        with patch('services.time.time', return_value=1234567890):
            # Эмулируем вызов on_message
            def on_message(ws, message):
                provider._on_message(ws, message)

            # Создаем mock WebSocket и вызываем on_message
            mock_ws = Mock()
            on_message(mock_ws, test_message)

        # Проверяем вызов callback с правильными данными
        mock_callback.assert_called_once_with(50123.45, 1234567890)

    def test_websocket_error_handling(self):
        """Тест обработки ошибок WebSocket."""
        provider = BinancePriceProvider()
        mock_callback = Mock()
        provider.callback = mock_callback
        provider.running = True

        test_error = "Connection error"

        # Эмулируем вызов on_error
        provider._on_error(None, test_error)

        # Проверяем вызов callback с ошибкой
        mock_callback.assert_called_once_with(None, pytest.approx(time.time(), rel=1), error=test_error)

    def test_websocket_close_handling(self):
        """Тест обработки закрытия WebSocket."""
        provider = BinancePriceProvider()
        mock_callback = Mock()
        provider.callback = mock_callback
        provider.running = True

        # Эмулируем вызов on_close
        provider._on_close(None, 1000, "Normal closure")

        # Проверяем вызов callback с флагом disconnected
        mock_callback.assert_called_once_with(None, pytest.approx(time.time(), rel=1), disconnected=True)


class TestNewsService:
    """Тесты для сервиса новостей."""

    def test_initialization(self):
        """Тест инициализации сервиса новостей."""
        service = NewsService("test_api_key")

        assert service.api_key == "test_api_key"
        assert service.last_update == 0
        assert service.cache == []

    @patch('services.requests.get')
    def test_fetch_news_success(self, mock_get):
        """Тест успешного получения новостей."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'articles': [
                {'title': 'Bitcoin reaches new all-time high'},
                {'title': 'Ethereum upgrade scheduled for next month'},
                {'title': '[Removed]'},  # Должен быть отфильтрован
                {'title': 'x' * 100}  # Должен быть обрезан
            ]
        }
        mock_get.return_value = mock_response

        service = NewsService("test_api_key")
        news = service.fetch_news()

        assert len(news) == 2
        assert "Bitcoin reaches new all-time high" in news
        assert "Ethereum upgrade" in news
        assert "[Removed]" not in news

        # Проверяем параметры запроса
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert "https://newsapi.org/v2/everything" in args[0]
        assert kwargs['params']['apiKey'] == "test_api_key"
        assert kwargs['params']['language'] == 'en'

    @patch('services.requests.get')
    def test_fetch_news_failure(self, mock_get):
        """Тест неудачного получения новостей."""
        mock_response = Mock()
        mock_response.status_code = 401  # Unauthorized
        mock_get.return_value = mock_response

        service = NewsService("invalid_key")
        news = service.fetch_news()

        # Должны вернуться заглушки
        assert len(news) == 5
        assert "Bitcoin" in news[0]

    @patch('services.requests.get')
    def test_fetch_news_exception(self, mock_get):
        """Тест исключения при получении новостей."""
        mock_get.side_effect = Exception("Network error")

        service = NewsService("test_key")
        news = service.fetch_news()

        # Должны вернуться заглушки
        assert len(news) == 5
        assert "Bitcoin" in news[0]

    def test_fetch_news_without_api_key(self):
        """Тест получения новостей без API ключа."""
        service = NewsService("YOUR_NEWS_API_KEY")
        news = service.fetch_news()

        # Должны вернуться заглушки
        assert len(news) == 5
        assert "Bitcoin" in news[0]

    def test_sample_news(self):
        """Тест заглушек новостей."""
        service = NewsService("test_key")
        sample_news = service._get_sample_news()

        assert len(sample_news) == 5
        assert all(isinstance(item, str) for item in sample_news)
        assert "Bitcoin" in sample_news[0]
        assert "Ethereum" in sample_news[1]


class TestNotificationService:
    """Тесты для сервиса уведомлений."""

    def test_initialization(self):
        """Тест инициализации сервиса уведомлений."""
        service = NotificationService()

        # Проверяем, что атрибуты инициализированы
        assert hasattr(service, 'has_sound')
        assert hasattr(service, 'has_notifications')

    @patch('services.winsound.MessageBeep')
    def test_play_alert_windows(self, mock_beep):
        """Тест воспроизведения звука на Windows."""
        service = NotificationService()
        service.has_sound = True
        service._play_sound = mock_beep

        service.play_alert()

        mock_beep.assert_called_once()

    @patch('builtins.print')
    def test_play_alert_fallback(self, mock_print):
        """Тест fallback воспроизведения звука."""
        service = NotificationService()
        service.has_sound = True
        service._play_sound = lambda: print('\a', end='', flush=True)

        service.play_alert()

        mock_print.assert_called_once_with('\a', end='', flush=True)

    def test_play_alert_no_sound(self):
        """Тест воспроизведения звука когда звук отключен."""
        service = NotificationService()
        service.has_sound = False

        # Не должно быть исключений
        service.play_alert()

    @patch('services.notification.notify')
    def test_show_notification_success(self, mock_notify):
        """Тест успешного показа уведомления."""
        service = NotificationService()
        service.has_notifications = True
        service._show_notification = mock_notify

        service.show_notification("Test Title", "Test Message", 10)

        mock_notify.assert_called_once_with(
            title="Test Title",
            message="Test Message",
            app_name="Crypto Tracker",
            timeout=10
        )

    def test_show_notification_no_support(self):
        """Тест показа уведомления когда они не поддерживаются."""
        service = NotificationService()
        service.has_notifications = False

        # Не должно быть исключений
        service.show_notification("Test Title", "Test Message")

    @patch('services.notification.notify')
    def test_show_notification_exception(self, mock_notify):
        """Тест исключения при показе уведомления."""
        mock_notify.side_effect = Exception("Notification error")

        service = NotificationService()
        service.has_notifications = True
        service._show_notification = mock_notify

        # Не должно быть исключений
        service.show_notification("Test Title", "Test Message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])