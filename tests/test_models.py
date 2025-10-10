"""Модульные тесты для моделей данных."""
import pytest
import time
from unittest.mock import Mock, patch
from collections import deque

from models import PriceMonitor, CurrencyConverter, AlertConfig, PriceUpdate


class TestPriceMonitor:
    """Тесты для класса PriceMonitor."""

    def test_initialization(self):
        """Тест инициализации монитора цен."""
        monitor = PriceMonitor("btcusdt")
        assert monitor.symbol == "btcusdt"
        assert monitor.running is True
        assert isinstance(monitor.price_history, deque)
        assert monitor.last_alert_time == 0
        assert monitor.last_price is None

    def test_add_valid_price(self):
        """Тест добавления валидной цены."""
        monitor = PriceMonitor("btcusdt")
        mock_observer = Mock()
        monitor.add_observer(mock_observer)

        test_price = 50000.0
        monitor.add_price(test_price)

        assert len(monitor.price_history) == 1
        assert monitor.last_price == test_price
        mock_observer.assert_called_once_with('price_update', {
            'price': test_price,
            'timestamp': pytest.approx(time.time(), rel=1),
            'symbol': 'btcusdt'
        })

    def test_add_invalid_prices(self):
        """Тест добавления невалидных цен."""
        monitor = PriceMonitor("btcusdt")

        # Отрицательная цена
        monitor.add_price(-100.0)
        assert len(monitor.price_history) == 0

        # Бесконечность
        monitor.add_price(float('inf'))
        assert len(monitor.price_history) == 0

        # NaN
        monitor.add_price(float('nan'))
        assert len(monitor.price_history) == 0

        # Ноль
        monitor.add_price(0.0)
        assert len(monitor.price_history) == 0

    def test_price_alert_up_threshold(self):
        """Тест оповещения при росте цены."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(threshold_up=1.0, threshold_down=1.0, cooldown_sec=0)

        # Добавляем начальную цену
        base_time = time.time()
        monitor.add_price(100.0)

        # Добавляем цену с ростом > 1%
        monitor.add_price(101.5)

        should_alert, message = monitor.check_alert(101.5, base_time + 1, config)
        assert should_alert is True
        assert "1.50%" in message
        assert "▲" in message

    def test_price_alert_down_threshold(self):
        """Тест оповещения при падении цены."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(threshold_up=1.0, threshold_down=1.0, cooldown_sec=0)

        # Добавляем начальную цену
        base_time = time.time()
        monitor.add_price(100.0)

        # Добавляем цену с падением > 1%
        monitor.add_price(98.5)

        should_alert, message = monitor.check_alert(98.5, base_time + 1, config)
        assert should_alert is True
        assert "1.50%" in message
        assert "▼" in message

    def test_absolute_alert_up(self):
        """Тест абсолютного оповещения сверху."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(
            threshold_up=1.0,
            threshold_down=1.0,
            absolute_up=70000.0,
            cooldown_sec=0
        )

        monitor.add_price(69000.0)
        should_alert, message = monitor.check_alert(70000.0, time.time(), config)

        assert should_alert is True
        assert "Цена ≥ 70000" in message

    def test_absolute_alert_down(self):
        """Тест абсолютного оповещения снизу."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(
            threshold_up=1.0,
            threshold_down=1.0,
            absolute_down=60000.0,
            cooldown_sec=0
        )

        monitor.add_price(61000.0)
        should_alert, message = monitor.check_alert(60000.0, time.time(), config)

        assert should_alert is True
        assert "Цена ≤ 60000" in message

    def test_alert_cooldown(self):
        """Тест коголдауна между оповещениями."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(threshold_up=1.0, threshold_down=1.0, cooldown_sec=10)

        base_time = time.time()
        monitor.add_price(100.0)
        monitor.add_price(102.0)  # +2%

        # Первое оповещение должно сработать
        should_alert, _ = monitor.check_alert(102.0, base_time, config)
        assert should_alert is True

        # Второе оповещение не должно сработать из-за коголдауна
        should_alert, _ = monitor.check_alert(103.0, base_time + 5, config)
        assert should_alert is False

        # После коголдауна должно снова сработать
        should_alert, _ = monitor.check_alert(104.0, base_time + 11, config)
        assert should_alert is True

    def test_no_alert_with_insufficient_history(self):
        """Тест отсутствия оповещения при недостаточной истории."""
        monitor = PriceMonitor("btcusdt")
        config = AlertConfig(threshold_up=1.0, threshold_down=1.0, cooldown_sec=0)

        # Только одна точка в истории
        monitor.add_price(100.0)

        should_alert, message = monitor.check_alert(102.0, time.time(), config)
        assert should_alert is False
        assert message == ""

    def test_stop_monitoring(self):
        """Тест остановки мониторинга."""
        monitor = PriceMonitor("btcusdt")
        monitor.running = False

        # После остановки цены не должны добавляться
        monitor.add_price(100.0)
        assert len(monitor.price_history) == 0


class TestAlertConfig:
    """Тесты для конфигурации оповещений."""

    def test_default_config(self):
        """Тест конфигурации по умолчанию."""
        config = AlertConfig(threshold_up=2.0, threshold_down=1.5)

        assert config.threshold_up == 2.0
        assert config.threshold_down == 1.5
        assert config.absolute_up is None
        assert config.absolute_down is None
        assert config.cooldown_sec == 15

    def test_config_with_absolute_values(self):
        """Тест конфигурации с абсолютными значениями."""
        config = AlertConfig(
            threshold_up=1.0,
            threshold_down=1.0,
            absolute_up=70000.0,
            absolute_down=60000.0,
            cooldown_sec=30
        )

        assert config.absolute_up == 70000.0
        assert config.absolute_down == 60000.0
        assert config.cooldown_sec == 30


class TestCurrencyConverter:
    """Тесты для конвертера валют."""

    def test_initialization(self):
        """Тест инициализации конвертера."""
        converter = CurrencyConverter()

        assert "USD" in converter.rates
        assert "RUB" in converter.rates
        assert converter.rates["USD"] == 1.0
        assert converter.last_update == 0

    @patch('models.requests.get')
    def test_update_rates_success(self, mock_get):
        """Тест успешного обновления курсов."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'price': '95.5'}
        mock_get.return_value = mock_response

        converter = CurrencyConverter()
        result = converter.update_rates()

        assert result is True
        assert converter.rates["RUB"] == 95.5
        assert converter.last_update > 0

    @patch('models.requests.get')
    def test_update_rates_failure(self, mock_get):
        """Тест неудачного обновления курсов."""
        mock_get.side_effect = Exception("Network error")

        converter = CurrencyConverter()
        original_rate = converter.rates["RUB"]
        result = converter.update_rates()

        assert result is False
        assert converter.rates["RUB"] == original_rate

    def test_get_usdt_rub_with_update(self):
        """Тест получения курса USDT/RUB с обновлением."""
        converter = CurrencyConverter()
        converter.last_update = 0  # Принудительно устаревшие данные

        with patch.object(converter, 'update_rates') as mock_update:
            rate = converter.get_usdt_rub()

            mock_update.assert_called_once()
            assert rate == converter.rates["RUB"]

    def test_get_usdt_rub_without_update(self):
        """Тест получения курса USDT/RUB без обновления."""
        converter = CurrencyConverter()
        converter.last_update = time.time()  # Свежие данные

        with patch.object(converter, 'update_rates') as mock_update:
            rate = converter.get_usdt_rub()

            mock_update.assert_not_called()
            assert rate == converter.rates["RUB"]

    def test_convert_currencies(self):
        """Тест конвертации между валютами."""
        converter = CurrencyConverter()
        converter.rates = {"USD": 1.0, "RUB": 95.0, "EUR": 0.92}

        # USD to RUB
        result = converter.convert(100, "USD", "RUB")
        assert result == 9500.0

        # RUB to USD
        result = converter.convert(9500, "RUB", "USD")
        assert result == 100.0

        # Same currency
        result = converter.convert(100, "USD", "USD")
        assert result == 100.0

        # Unknown currency
        result = converter.convert(100, "USD", "UNKNOWN")
        assert result is None


class TestPriceUpdate:
    """Тесты для модели обновления цены."""

    def test_price_update_creation(self):
        """Тест создания объекта обновления цены."""
        timestamp = time.time()
        price_update = PriceUpdate(
            timestamp=timestamp,
            price=50000.0,
            volume=1000.0,
            symbol="btcusdt"
        )

        assert price_update.timestamp == timestamp
        assert price_update.price == 50000.0
        assert price_update.volume == 1000.0
        assert price_update.symbol == "btcusdt"

    def test_price_update_defaults(self):
        """Тест значений по умолчанию."""
        price_update = PriceUpdate(timestamp=time.time(), price=50000.0)

        assert price_update.volume == 0.0
        assert price_update.symbol == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])