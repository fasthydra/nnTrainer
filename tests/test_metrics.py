import pytest
import time
import torch
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

    # Имитация обработки нескольких батчей
    for _ in range(2):
        metrics_logger.start_batch()
        # Имитация задержки времени обработки батча
        time.sleep(0.01)
        metrics_logger.end_batch(
            outputs=torch.randn(10, 2),  # Примерные выходные данные модели
            labels=torch.randint(0, 2, (10,)),  # Примерные метки
            loss=torch.tensor(0.5),  # Примерное значение потерь
            batch_size=10  # Размер батча
        )

    # Имитация окончания эпохи
    mode = "training"
    metrics_logger.end_epoch(mode)

    # Проверка обновления метрик эпохи
    assert mode in metrics_logger.epoch_metrics, "Метрики эпохи обучения не были обновлены"
    epoch_metrics = metrics_logger.epoch_metrics[mode]

    assert "epoch" in epoch_metrics and epoch_metrics["epoch"] == 1, "Неправильный номер эпохи"
    assert "duration" in epoch_metrics, "Отсутствует метрика продолжительности"
    assert "batches" in epoch_metrics and len(epoch_metrics["batches"]) == 2, "Неправильное количество метрик батчей"
    assert "total" in epoch_metrics, "Отсутствует секция общих метрик"

    # Проверка корректности расчета средних значений метрик
    for key in epoch_metrics["total"]:
        average_metric = epoch_metrics[key]
        assert average_metric == epoch_metrics["total"][key] / 20, "Некорректное среднее значение метрики"

def test_end_batch_error_handling():
    logger = MetricsLogger()

    # Создаем примерные данные для теста
    outputs = torch.randn(10, 2)
    labels = torch.randint(0, 2, (10,))
    invalid_loss = torch.randn(10, 2)  # Неверный формат loss

    # Проверяем, что вызывается исключение ValueError
    with pytest.raises(ValueError):
        logger.end_batch(outputs, labels, invalid_loss)

    # Проверяем, что исключение не вызывается при корректном формате loss
    try:
        valid_loss = torch.tensor(0.5)
        logger.end_batch(outputs, labels, valid_loss)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_end_batch_metrics_update_correctly():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Создаем примерные данные для теста
    outputs = torch.randn(10, 2)
    labels = torch.randint(0, 2, (10,))
    loss = torch.tensor(0.5)

    # Начало обработки батча
    logger.start_batch()

    # Имитация задержки времени обработки батча
    time.sleep(0.01)

    # Завершение обработки батча
    logger.end_batch(outputs, labels, loss)

    # Проверка обновления метрик батча
    assert len(logger.batches_metrics) == 1, "Метрики батча не были добавлены"
    batch_metrics = logger.batches_metrics[0]

    assert "duration" in batch_metrics, "Отсутствует метрика продолжительности"
    assert "metrics" in batch_metrics and "loss" in batch_metrics["metrics"], "Отсутствует метрика потерь"
    assert batch_metrics["metrics"]["loss"] == loss.item(), "Некорректное значение потерь"
    assert "dummy_metric" in batch_metrics["metrics"], "Отсутствует пользовательская метрика"
    assert batch_metrics["metrics"][
               "dummy_metric"] == outputs.mean().item(), "Некорректное значение пользовательской метрики"

def test_start_epoch_resets_metrics_correctly():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики и имитируем некоторые метрики батча
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")
    logger.batches_metrics = [{"metrics": {"dummy_metric": 0.5}}]
    logger.total_metrics = {"dummy_metric": 0.5}

    # Имитируем начало новой эпохи
    logger.start_epoch(2)

    # Проверка сброса метрик
    assert logger._current_epoch == 2, "Номер эпохи не был обновлен"
    assert logger._processed_data == 0, "Счетчик обработанных данных не сброшен"
    assert logger.batches_metrics == [], "Метрики батчей не сброшены"
    assert all(value == 0.0 for value in logger.total_metrics.values()), "Общие метрики не сброшены"
    assert logger.epoch_metrics == {"training": {}, "validation": {}, "testing": {}}, "Метрики эпох не сброшены"

def test_end_epoch_calculates_total_metrics_correctly():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики и имитируем метрики батча
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация обработки нескольких батчей
    for i in range(2):
        outputs = torch.randn(10, 2)
        labels = torch.randint(0, 2, (10,))
        loss = torch.tensor(0.5)
        logger.start_batch()
        time.sleep(0.01)  # Имитация задержки времени обработки батча
        logger.end_batch(outputs, labels, loss)

    # Завершение эпохи
    mode = "training"
    logger.end_epoch(mode)

    # Проверка правильности расчета общих метрик
    assert mode in logger.epoch_metrics, "Метрики эпохи не были обновлены"
    epoch_metrics = logger.epoch_metrics[mode]

    assert "total" in epoch_metrics, "Отсутствует секция общих метрик"
    total_metrics = epoch_metrics["total"]

    expected_dummy_metric_total = sum(
        batch["metrics"]["dummy_metric"] * batch["batch_size"] for batch in logger.batches_metrics)
    assert total_metrics[
               "dummy_metric"] == expected_dummy_metric_total, "Некорректное значение общей пользовательской метрики"

def test_metric_function_functionality():
    logger = MetricsLogger()

    # Создаем моковую функцию метрики
    mock_metric_function = Mock(return_value=0.7)

    logger.add_metric_function("mock_metric", mock_metric_function, "batch")

    # Имитация обработки батча
    outputs = torch.randn(10, 2)
    labels = torch.randint(0, 2, (10,))
    loss = torch.tensor(0.5)
    batch_size = 10

    logger.start_batch()
    logger.end_batch(outputs, labels, loss, batch_size=batch_size)

    # Проверка вызова функции метрики с правильными аргументами
    mock_metric_function.assert_called_once_with(outputs, labels)

    # Проверка корректности влияния результата функции на метрики
    assert logger.batches_metrics[0]["metrics"]["mock_metric"] == 0.7, "Некорректное значение пользовательской метрики в метриках батча"
    assert logger.total_metrics["mock_metric"] == 0.7 * batch_size, "Некорректное значение в общих метриках"

def test_metrics_across_multiple_epochs():
    logger = MetricsLogger()

    # Создаем моковую функцию метрики
    mock_metric_function = Mock(return_value=0.7)
    logger.add_metric_function("mock_metric", mock_metric_function, "batch")

    final_epoch = 3
    # Проведем несколько эпох
    for epoch in range(1, final_epoch + 1):
        logger.start_epoch(epoch)

        # Имитация обработки нескольких батчей в эпохе
        for _ in range(2):
            outputs = torch.randn(10, 2)
            labels = torch.randint(0, 2, (10,))
            loss = torch.tensor(0.5)
            batch_size = 10

            logger.start_batch()
            logger.end_batch(outputs, labels, loss, batch_size=batch_size)

        logger.end_epoch("training")

    # Проверка метрик последней эпохи
    assert logger.epoch_metrics["training"]["epoch"] == final_epoch, "Некорректный номер последней эпохи"
    assert "mock_metric" in logger.epoch_metrics["training"]["total"], "Отсутствует пользовательская метрика в последней эпохе"


def test_add_metric_function_with_invalid_type():
    logger = MetricsLogger()

    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    invalid_metric_type = "invalid_type"

    # Попытка добавить метрику с неверным типом
    with pytest.raises(ValueError):
        logger.add_metric_function("dummy_metric", dummy_metric, invalid_metric_type)

def test_end_batch_with_varying_metrics():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return outputs.sum().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация обработки нескольких батчей с разными метриками
    for i in range(2):
        outputs = torch.randn(10, 2) * (i + 1)  # Создаем разные данные для каждого батча
        labels = torch.randint(0, 2, (10,))
        loss = torch.tensor(0.5 * (i + 1))
        batch_size = 10

        logger.start_batch()
        logger.end_batch(outputs, labels, loss, batch_size=batch_size)

    # Проверяем, что метрики батча обновлены правильно
    assert len(logger.batches_metrics) == 2, "Неверное количество метрик батчей"
    assert logger.total_metrics["dummy_metric"] == sum(batch["metrics"]["dummy_metric"] * batch["batch_size"] for batch in logger.batches_metrics), "Некорректное значение общей пользовательской метрики"

def test_end_epoch_with_no_metrics():
    logger = MetricsLogger()

    # Начало новой эпохи
    logger.start_epoch(1)

    # Завершение эпохи без добавления или обновления каких-либо метрик
    logger.end_epoch("training")

    # Проверка, что метрики эпохи обновлены, даже если никакие метрики не были добавлены
    assert "training" in logger.epoch_metrics, "Метрики эпохи обучения не были обновлены"
    assert "epoch" in logger.epoch_metrics["training"] and logger.epoch_metrics["training"]["epoch"] == 1, "Неверный номер эпохи"
    assert "duration" in logger.epoch_metrics["training"] and logger.epoch_metrics["training"]["duration"] >= 0, "Неверное значение продолжительности"
    assert logger.epoch_metrics["training"]["batches"] == [], "Список метрик батчей должен быть пустым"
    assert logger.epoch_metrics["training"]["total"] == {}, "Словарь общих метрик должен быть пустым"

def test_end_batch_with_empty_batches():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return 0.0 if outputs.nelement() == 0 else outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация обработки пустого батча
    outputs = torch.tensor([])
    labels = torch.tensor([])
    loss = torch.tensor(0.0)
    batch_size = 0

    logger.start_batch()
    logger.end_batch(outputs, labels, loss, batch_size=batch_size)

    # Проверка корректности обработки пустого батча
    assert len(logger.batches_metrics) == 1, "Метрики батча должны быть обновлены даже для пустого батча"
    assert logger.batches_metrics[0]["metrics"]["dummy_metric"] == 0.0, "Некорректное значение метрики для пустого батча"

def test_metric_function_handling_invalid_output():
    logger = MetricsLogger()

    # Создаем метрическую функцию, которая вызывает исключение
    def faulty_metric_function(outputs, labels):
        raise RuntimeError("Ошибка при вычислении метрики")

    logger.add_metric_function("faulty_metric", faulty_metric_function, "batch")

    # Создаем метрическую функцию, возвращающую None
    def none_metric_function(outputs, labels):
        return None

    logger.add_metric_function("none_metric", none_metric_function, "batch")

    # Имитация обработки батча
    outputs = torch.randn(10, 2)
    labels = torch.randint(0, 2, (10,))
    loss = torch.tensor(0.5)
    batch_size = 10

    logger.start_batch()
    logger.end_batch(outputs, labels, loss, batch_size=batch_size)

    # Проверка, что метрика "faulty_metric" не была обновлена из-за ошибки
    assert "faulty_metric" not in logger.batches_metrics[0]["metrics"], "Метрика 'faulty_metric' не должна обновляться при ошибке"

    # Проверка, что метрика "none_metric" не была обновлена из-за возвращения None
    assert "none_metric" not in logger.batches_metrics[0]["metrics"], "Метрика 'none_metric' не должна обновляться при возвращении None"

    # Проверка, что обработка None не вызывает исключений
    try:
        logger.start_batch()
        logger.end_batch(outputs, labels, loss, batch_size=batch_size)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

def test_last_epoch_metrics_saving():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация нескольких эпох обучения
    final_epoch = 3
    for epoch in range(1, final_epoch + 1):
        logger.start_epoch(epoch)
        for _ in range(2):  # Имитация двух батчей в каждой эпохе
            outputs = torch.randn(10, 2)
            labels = torch.randint(0, 2, (10,))
            loss = torch.tensor(0.5)
            batch_size = 10

            logger.start_batch()
            logger.end_batch(outputs, labels, loss, batch_size=batch_size)

        logger.end_epoch("training")

    # Проверка, что сохранены метрики только последней эпохи
    assert logger.epoch_metrics["training"]["epoch"] == final_epoch, "Неверно сохранена информация о последней эпохе"

import time

def test_timing_metrics():
    logger = MetricsLogger()

    # Начало эпохи
    logger.start_epoch(1)
    time.sleep(0.01)  # Имитация задержки времени

    # Начало и конец батча
    logger.start_batch()
    time.sleep(0.01)  # Имитация обработки батча
    logger.end_batch(torch.randn(10, 2), torch.randint(0, 2, (10,)), torch.tensor(0.5), 10)

    # Конец эпохи
    logger.end_epoch("training")

    # Проверка временных метрик
    batch_metrics = logger.batches_metrics[0]
    epoch_metrics = logger.epoch_metrics["training"]

    assert batch_metrics["duration"] > 0, "Продолжительность батча должна быть положительной"
    assert epoch_metrics["duration"] > batch_metrics["duration"], "Продолжительность эпохи должна быть больше продолжительности батча"

def test_mode_switching():
    logger = MetricsLogger()

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация эпохи обучения
    logger.start_epoch(1)
    logger.end_batch(torch.randn(10, 2), torch.randint(0, 2, (10,)), torch.tensor(0.5), 10)
    logger.end_epoch("training")

    # Проверка метрик после обучения
    assert "training" in logger.epoch_metrics, "Отсутствуют метрики обучения"

    # Имитация эпохи валидации
    logger.start_epoch(1)
    logger.end_batch(torch.randn(10, 2), torch.randint(0, 2, (10,)), torch.tensor(0.5), 10)
    logger.end_epoch("validation")

    # Проверка метрик после валидации
    assert "validation" in logger.epoch_metrics, "Отсутствуют метрики валидации"

import torch.utils.data as data

def test_integration_with_dataloader():
    logger = MetricsLogger()

    # Создаем простой DataLoader
    dataset = data.TensorDataset(torch.randn(20, 2), torch.randint(0, 2, (20,)))
    data_loader = data.DataLoader(dataset, batch_size=10)

    # Добавляем пользовательскую функцию метрики
    def dummy_metric(outputs, labels):
        return outputs.mean().item()

    logger.add_metric_function("dummy_metric", dummy_metric, "batch")

    # Имитация эпохи обучения с использованием DataLoader
    logger.start_epoch(1)
    for inputs, targets in data_loader:
        outputs = inputs  # В реальном сценарии здесь будет вызов модели
        loss = torch.tensor(0.5)  # Примерное значение потерь
        logger.end_batch(outputs, targets, loss, batch_size=inputs.size(0))

    logger.end_epoch("training")

    # Проверка метрик после обработки данных из DataLoader
    assert len(logger.batches_metrics) == len(data_loader), "Неверное количество метрик батчей"
    assert "dummy_metric" in logger.batches_metrics[0]["metrics"], "Отсутствует пользовательская метрика"
