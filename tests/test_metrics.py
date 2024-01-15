import pytest
from metrics import MetricsLogger
from unittest.mock import Mock

@pytest.fixture
def metrics_logger():
    return MetricsLogger()

def test_initialization(metrics_logger):
    assert isinstance(metrics_logger.epoch_metrics, dict), "epoch_metrics должен быть словарем"
    assert isinstance(metrics_logger.batches_metrics, list), "batches_metrics должен быть списком"
    assert isinstance(metrics_logger.total_metrics, dict), "total_metrics должен быть словарем"
    assert metrics_logger.epoch_metrics == {"training": {}, "validation": {}, "testing": {}}, "Некорректные начальные значения для epoch_metrics"
    assert metrics_logger.batches_metrics == [], "Некорректные начальные значения для batches_metrics"
    assert metrics_logger.total_metrics == {}, "Некорректные начальные значения для total_metrics"


def test_add_metric_function(metrics_logger):
    def dummy_metric_function(outputs, labels):
        return 0.0

    metrics_logger.add_metric_function("dummy_metric", dummy_metric_function, "batch")
    assert "dummy_metric" in metrics_logger.batch_metric_functions, "Функция метрики не была добавлена для batch"
    assert callable(metrics_logger.batch_metric_functions["dummy_metric"]), "Добавленная функция метрики должна быть вызываемой"

    metrics_logger.add_metric_function("dummy_epoch_metric", dummy_metric_function, "epoch")
    assert "dummy_epoch_metric" in metrics_logger.total_metric_functions, "Функция метрики не была добавлена для epoch"
    assert callable(metrics_logger.total_metric_functions["dummy_epoch_metric"]), "Добавленная функция метрики должна быть вызываемой"


@pytest.fixture
def mock_data_loader():
    # Создаем мок DataLoader'а
    mock_loader = Mock()
    # Здесь можно настроить поведение mock_loader, если это необходимо для теста
    return mock_loader


def test_calculate_epoch_metrics(metrics_logger, mock_data_loader):
    # Создаем моковые функции метрик
    mock_metric_function1 = Mock(return_value=0.5)
    mock_metric_function2 = Mock(return_value=0.8)

    # Добавляем моковые функции метрик
    metrics_logger.add_metric_function("mock_metric1", mock_metric_function1, "epoch")
    metrics_logger.add_metric_function("mock_metric2", mock_metric_function2, "epoch")

    # Вызываем calculate_epoch_metrics и проверяем результаты
    metrics_logger.calculate_epoch_metrics(mock_data_loader, "training")

    # Убеждаемся, что моковые функции метрик были вызваны
    mock_metric_function1.assert_called_once_with(mock_data_loader)
    mock_metric_function2.assert_called_once_with(mock_data_loader)

    # Проверяем, что результаты метрик сохранены корректно
    assert metrics_logger.epoch_metrics["training"]["mock_metric1"] == 0.5, "Некорректное значение для mock_metric1"
    assert metrics_logger.epoch_metrics["training"]["mock_metric2"] == 0.8, "Некорректное значение для mock_metric2"


def test_start_epoch_updates_attributes_correctly(metrics_logger):
    metrics_logger.start_epoch(1)

    assert metrics_logger._current_epoch == 1, "Текущая эпоха не обновилась правильно"
    assert isinstance(metrics_logger._epoch_start_time, float), "Время начала эпохи не установлено"
    assert metrics_logger._processed_data == 0, "Счетчик обработанных данных не сброшен"
    assert metrics_logger.batches_metrics == [], "Метрики батчей не сброшены"
    assert all(value == 0.0 for value in metrics_logger.total_metrics.values()), "Общие метрики не сброшены"
    assert metrics_logger._batch_sizes == [], "Атрибут _batch_sizes не был сброшен"
    assert metrics_logger._epoch_start_time != 0.0, "Время начала эпохи (_epoch_start_time) не было отмечено"

def test_end_epoch_updates_metrics_correctly(metrics_logger):
    # Имитация начала эпохи
    metrics_logger.start_epoch(1)

    # Имитация добавления метрик батчей (обычно это делается в процессе обучения)
    metrics_logger.batches_metrics = [
        {"loss": 0.8, "accuracy": 0.7},
        {"loss": 0.6, "accuracy": 0.8}
    ]

    # Имитация окончания эпохи
    metrics_logger.end_epoch("training")

    # Проверки
    assert "training" in metrics_logger.epoch_metrics, "Метрики эпохи обучения не были обновлены"
    assert isinstance(metrics_logger.epoch_metrics["training"], dict), "Метрики эпохи обучения должны быть словарем"

    # Проверка наличия основных ключей в метриках эпохи
    training_metrics = metrics_logger.epoch_metrics["training"]
    assert "duration" in training_metrics, "Должен быть ключ duration"
    assert "batches" in training_metrics, "Должен быть ключ batches"
    assert "total" in training_metrics, "Должен быть ключ total"

    # Проверка корректности собранных метрик
    assert len(training_metrics["batches"]) == 2, "Неверное количество метрик батчей"
    assert training_metrics["total"]["loss"] == 1.4, "Неверное значение суммарного loss"
    assert training_metrics["total"]["accuracy"] == 1.5, "Неверное значение суммарной accuracy"

    # Проверка расчета средних значений метрик (при необходимости)
    # ...
