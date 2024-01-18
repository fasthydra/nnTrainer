import copy
import tempfile
import shutil
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from nn_trainer import ModelTrainer
from nn_trainer import TrainingProgressStorage

import unittest
from unittest.mock import Mock


def set_random_seeds(seed_value=0):
    """Устанавливает random seed для обеспечения воспроизводимости результатов."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # для использования в многокартовых средах
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # Два входных признака и один выход

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Применяем сигмоиду для бинарной классификации


def test_model_trainer_initialization():
    mock_model = Mock(spec=torch.nn.Module)
    mock_optimizer = Mock(spec=torch.optim.Optimizer)
    mock_scheduler = Mock(spec=torch.optim.lr_scheduler._LRScheduler)
    mock_criterion = Mock(spec=torch.nn.modules.loss._Loss)

    trainer = ModelTrainer(
        model=mock_model,
        optimizer=mock_optimizer,
        scheduler=mock_scheduler,
        criterion=mock_criterion,
        device='cpu',
        save_every_k_epochs=2
    )

    # Проверка инициализации основных компонентов
    assert trainer.model is mock_model, "Модель не инициализирована корректно"
    assert trainer.optimizer is mock_optimizer, "Оптимизатор не инициализирован корректно"
    assert trainer.scheduler is mock_scheduler, "Планировщик не инициализирован корректно"
    assert trainer.criterion is mock_criterion, "Критерий не инициализирован корректно"
    assert trainer.device == 'cpu', "Устройство не инициализировано корректно"
    assert trainer.save_every_k_epochs == 2, "Неверная инициализация параметра сохранения каждые k эпох"

    # Проверка инициализации вспомогательных компонентов
    assert trainer.metrics_logger is not None, "Logger метрик не инициализирован"
    assert trainer.history == [], "История обучения не инициализирована корректно"


def test_training_process_with_history():
    set_random_seeds(0)

    # Использование реальной модели, оптимизатора, планировщика и критерия
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.BCELoss()

    # Создание тестовых данных и DataLoader
    test_inputs = torch.randn(10, 2)
    test_targets = torch.randint(0, 2, (10, 1)).float()
    test_dataset = data.TensorDataset(test_inputs, test_targets)
    test_data_loader = data.DataLoader(test_dataset, batch_size=5)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device='cpu'
    )

    # Сохранение начального состояния модели
    initial_model_state = copy.deepcopy(model.state_dict())

    # Имитация вызова метода train
    trainer.train(test_data_loader, test_data_loader, 2)

    current_model_state = model.state_dict()

    # Проверки вызовов и состояния после обучения
    assert len(trainer.history) == 2, "История обучения должна содержать записи за каждую эпоху"

    # Проверка изменения состояния модели
    for param in initial_model_state:
        assert not torch.equal(initial_model_state[param],
                               current_model_state[param]), "Параметр модели не изменился: " + param

    assert all('training' in epoch_info for epoch_info in
               trainer.history), "Должны быть сохранены метрики обучения для каждой эпохи"
    assert all('validation' in epoch_info for epoch_info in
               trainer.history), "Должны быть сохранены метрики валидации для каждой эпохи"

    # Проверка наличия метрик обучения и валидации для каждой эпохи
    for i, epoch_history in enumerate(trainer.history):

        # Проверки для обучения
        assert 'training' in epoch_history, "Должны быть сохранены метрики обучения"
        assert epoch_history['training']['epoch'] == i + 1, f"Номер эпохи должен быть {i + 1}"
        assert len(epoch_history['training']['batches']) == len(
            test_data_loader), "Неверное количество батчей в обучении"
        assert 'total' in epoch_history['training'], "Должны быть сохранены итоговые метрики обучения"
        assert 'loss' in epoch_history['training']['total'], "Должны быть сохранены потери обучения"
        assert epoch_history['training']['total']['loss'] is not None, "Потери обучения не должны быть пустыми"

        # Проверки для валидации
        assert 'validation' in epoch_history, "Должны быть сохранены метрики валидации"
        assert epoch_history['validation']['epoch'] == i + 1, f"Номер эпохи должен быть {i + 1}"
        assert len(epoch_history['validation']['batches']) == len(
            test_data_loader), "Неверное количество батчей в валидации"
        assert 'total' in epoch_history['validation'], "Должны быть сохранены итоговые метрики валидации"
        assert 'loss' in epoch_history['validation']['total'], "Должны быть сохранены потери валидации"
        assert epoch_history['validation']['total']['loss'] is not None, "Потери валидации не должны быть пустыми"

        # Общие проверки
        assert 'duration' in epoch_history['training'], "Должна быть сохранена продолжительность обучения"
        assert 'duration' in epoch_history['validation'], "Должна быть сохранена продолжительность валидации"


def test_eval_epoch():
    # Подготовка тестовых данных и модели
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.BCELoss()
    # metrics_logger = MetricsLogger()
    # storage = TrainingProgressStorage('test_dir')

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device='cpu',
        # metrics_logger=metrics_logger,
        # storage=storage
    )

    # Создание тестовых данных
    test_inputs = torch.randn(10, 2)
    test_targets = torch.randint(0, 2, (10, 1)).float()
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5)

    # Имитация начала эпохи, чтобы добавить элемент в self.history
    trainer._start_epoch(current_epoch=1, end_epoch=1)

    # Вызов метода _eval_epoch
    trainer._eval_epoch(test_data_loader, 1, mode='validation')

    # Проверки
    assert 'validation' in trainer.history[-1], "Должен быть ключ 'validation' в последней записи истории"
    assert isinstance(trainer.history[-1]['validation'], dict), "Значение 'validation' должно быть словарем"
    assert 'loss' in trainer.history[-1]['validation'], "Должен быть ключ 'loss' в данных валидации"
    assert trainer.history[-1]['validation']['loss'] is not None, "Значение потерь валидации не должно быть None"

    validation_metrics = trainer.history[-1]['validation']

    # Проверка наличия метрик для каждого батча
    assert 'batches' in validation_metrics, "Должны быть сохранены метрики для каждого батча"
    assert len(validation_metrics['batches']) == len(
        test_data_loader), "Количество батчей в метриках должно совпадать с количеством в DataLoader"

    # Проверка корректности сохранения общих метрик валидации
    assert 'total' in validation_metrics, "Должны быть сохранены итоговые метрики валидации"
    assert 'loss' in validation_metrics['total'], "В итоговых метриках должен быть ключ 'loss'"
    assert isinstance(validation_metrics['total']['loss'], float), "Итоговые потери должны быть числом"

    # Проверка корректности временных меток
    assert 'duration' in validation_metrics, "Должно быть сохранено время выполнения валидации"
    assert isinstance(validation_metrics['duration'], float), "Продолжительность должна быть числом"


class TestModelTrainer(unittest.TestCase):

    def test_call_callbacks(self):
        # Создание простой тестовой модели
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.BCELoss()

        # Инициализация ModelTrainer
        trainer = ModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            criterion=criterion,
            device=torch.device('cpu'),
        )

        # Создание mock callback функций
        mock_callback = Mock()

        # Установка mock callback функций в trainer
        trainer.callbacks = [mock_callback]

        # Тестирование вызова callback'ов
        trainer._call_callbacks('start_train')
        trainer._call_callbacks('end_train')

        # Проверка вызова mock callback'ов
        assert mock_callback.call_count == 2, "Callback должен быть вызван дважды"

        # Проверка вызова с правильными аргументами
        first_call, second_call = mock_callback.call_args_list
        assert first_call[0][0] == 'start_train', "Первый вызов должен быть для 'start_train'"
        assert second_call[0][0] == 'end_train', "Второй вызов должен быть для 'end_train'"


class ModelTrainerTestEarlyStop(unittest.TestCase):

    def setUp(self):
        # Создание тестовой модели и необходимых компонентов
        self.model = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = torch.nn.BCELoss()
        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            criterion=self.criterion,
            device=torch.device('cpu')
        )

    def test_early_stop(self):
        # Установка значения для early stop
        self.trainer._early_stop_best_val = 0.5

        # Проверка остановки при улучшении
        self.trainer.get_early_stop_value = Mock(return_value=0.4)
        self.trainer.early_stop(patience=3)
        assert self.trainer._early_stop_epochs_without_improvement == 0, "Должен сбросить счетчик при улучшении"

        # Проверка отсутствия остановки при отсутствии улучшений
        self.trainer.get_early_stop_value = Mock(return_value=0.6)
        for _ in range(2):
            self.trainer.early_stop(patience=3)
        assert self.trainer._early_stop_epochs_without_improvement == 2, "Должен увеличивать счетчик при отсутствии улучшений"

        # Проверка остановки после исчерпания терпения
        stop = self.trainer.early_stop(patience=3)
        assert stop, "Должен остановить обучение после исчерпания терпения"

        # Проверка сценария без ранней остановки (patience=None)
        self.trainer._early_stop_best_val = 0.5
        self.trainer.get_early_stop_value = Mock(return_value=0.6)
        stop = self.trainer.early_stop(patience=None)
        assert not stop, "Не должен останавливаться при patience=None"


class ModelTrainerTestRestore(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = torch.nn.BCELoss()

        # Создание временной директории для TrainingProgressStorage
        self.temp_dir = tempfile.mkdtemp()

        self.storage = TrainingProgressStorage(self.temp_dir)
        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            criterion=self.criterion,
            device=torch.device('cpu'),
            storage=self.storage,
            save_every_k_epochs=1
        )

    def test_save_and_restore(self):
        # Имитация изменения состояния модели и оптимизатора
        for param in self.model.parameters():
            param.data += torch.randn_like(param)

        # Сохранение текущего состояния
        self.trainer.save_progress(1)

        # Сохранение состояния после первого изменения для последующего сравнения
        saved_model_state = copy.deepcopy(self.model.state_dict())

        # Второе изменение состояния модели
        for param in self.model.parameters():
            param.data += torch.randn_like(param)

        # Восстановление сохраненного состояния
        self.trainer.restore()

        # Проверка восстановления состояния модели
        restored_model_state = self.model.state_dict()
        for saved_param, restored_param in zip(saved_model_state.values(), restored_model_state.values()):
            assert torch.allclose(saved_param, restored_param, atol=1e-6), "Состояние модели не было восстановлено правильно"

    def tearDown(self):
        # Очистка временной директории
        shutil.rmtree(self.temp_dir)


class ModelTrainerDataloader(unittest.TestCase):

    def setUp(self):
        # Инициализация модели, оптимизатора и критерия
        self.model = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = torch.nn.MSELoss()
        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            criterion=self.criterion,
            device=torch.device('cpu')
        )

    def test_data_loader_interaction(self):
        # Создание тестового DataLoader
        inputs = torch.randn(100, 10)
        targets = torch.randn(100, 2)
        dataset = TensorDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size=10)

        # Имитация начала эпохи, чтобы добавить элемент в self.history
        self.trainer._start_epoch(current_epoch=1, end_epoch=1)

        # Вызов метода обучения и валидации
        self.trainer._fit_epoch(data_loader, 1)
        self.trainer._eval_epoch(data_loader, 1)

        # Проверка, что метрики были корректно обновлены
        assert 'training' in self.trainer.history[-1], "Отсутствуют метрики обучения"
        assert 'validation' in self.trainer.history[-1], "Отсутствуют метрики валидации"

        # Проверка корректности обработки данных для обучения и валидации
        assert len(self.trainer.history[-1]['training']['batches']) == 10, "Неверное количество батчей в обучении"
        assert len(self.trainer.history[-1]['validation']['batches']) == 10, "Неверное количество батчей в валидации"


class ModelTrainerException(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.criterion = torch.nn.MSELoss()
        self.trainer = ModelTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            criterion=self.criterion,
            device=torch.device('cpu')
        )

    def test_exception_handling(self):
        # Создание DataLoader с некорректными данными
        inputs = torch.randn(10, 5)  # Неверная размерность входных данных
        targets = torch.randn(10, 2)
        dataset = TensorDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size=2)

        # Имитация начала эпохи
        self.trainer._start_epoch(current_epoch=1, end_epoch=1)

        # Тестирование обработки исключений при обучении
        with self.assertRaises(RuntimeError):
            self.trainer._fit_epoch(data_loader, 1)

        # Тестирование обработки исключений при валидации
        with self.assertRaises(RuntimeError):
            self.trainer._eval_epoch(data_loader, 1)
