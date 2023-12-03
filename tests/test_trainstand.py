import unittest
from unittest.mock import Mock, patch, ANY

import torch
import torch.nn as nn

from stand import TrainStand


class TestTrainStand(unittest.TestCase):
    def setUp(self):
        # Подготовка данных для теста
        self.datasets = {
            "train": torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100)),
            "valid": torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100)),
            "test": torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100))
        }
        self.max_epoch = 10
        self.optimizer_params = {"type": "Adam", "args": {"lr": 0.001}}
        self.scheduler_params = {"type": "StepLR", "args": {"step_size": 1, "gamma": 0.1}}

    def test_initialization(self):
        # Тестирование инициализации
        train_stand = TrainStand(
            self.datasets,
            self.max_epoch,
            self.optimizer_params,
            self.scheduler_params
        )
        self.assertIsNotNone(train_stand)
        self.assertEqual(train_stand.max_epoch, self.max_epoch)

    def test_default_device_initialization(self):
        train_stand = TrainStand(self.datasets, self.max_epoch, self.optimizer_params, self.scheduler_params)
        self.assertIsNotNone(train_stand.device)

    def test_initialization_with_invalid_dataset(self):
        with self.assertRaises(TypeError):
            TrainStand("not_a_dataset", self.max_epoch, self.optimizer_params, self.scheduler_params)

    def test_initialization_with_negative_max_epoch(self):
        with self.assertRaises(ValueError):
            TrainStand(self.datasets, -1, self.optimizer_params, self.scheduler_params)

    @patch('stand.TrainStand.run')
    def test_launch_with_valid_data(self, mock_run):
        models = {"model1": SimpleTestModel()}
        loss_functions = {"loss1": MockLossFunction()}
        learning_rates = [0.001, 0.01]
        batch_sizes = [32, 64]

        train_stand = TrainStand(self.datasets, self.max_epoch, self.optimizer_params, self.scheduler_params)
        train_stand.launch(models=models, loss_functions=loss_functions, learning_rates=learning_rates,
                           batch_sizes=batch_sizes)

        # Проверяем, был ли вызван метод run и сколько раз
        self.assertTrue(mock_run.called)
        self.assertEqual(len(mock_run.call_args_list),
                         len(models) * len(loss_functions) * len(learning_rates) * len(batch_sizes))

    @patch('stand.TrainStand.run')
    def test_launch_with_invalid_data(self, mock_run):
        models = {"invalid_model": None}
        loss_functions = {"invalid_loss": None}
        learning_rates = [0.001, 0.01]
        batch_sizes = [0, -1]  # Невалидные значения batch_size

        train_stand = TrainStand(self.datasets, self.max_epoch, self.optimizer_params, self.scheduler_params)

        # В этом случае, следует ожидать исключение или другую форму обработки ошибок
        with self.assertRaises(ValueError):
            train_stand.launch(models=models, loss_functions=loss_functions, learning_rates=learning_rates,
                               batch_sizes=batch_sizes)

        # Проверяем, что метод run не был вызван
        self.assertFalse(mock_run.called)

    @patch('training.ModelTrainer')
    @patch('training.TrainingProgressStorage')
    def test_run(self, mock_storage, mock_trainer):
        model = SimpleTestModel()
        loss_function = MockLossFunction()
        dataloaders = {"train": MockDataLoader(), "valid": MockDataLoader()}

        start_lr = 0.01
        max_epoch = 5

        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.train.return_value = "mock_history"
        mock_trainer_instance.restore.return_value = 0

        train_stand = TrainStand(self.datasets, self.max_epoch, self.optimizer_params, self.scheduler_params)
        history = train_stand.run("test_run", model, loss_function, dataloaders, start_lr, max_epoch)

        # Проверяем, были ли вызваны нужные методы
        mock_trainer.assert_called_once_with(
            model=model,
            optimizer=ANY,
            scheduler=ANY,
            criterion=loss_function,
            device=ANY,
            callbacks=ANY,
            save_every_k_epochs=ANY  # ANY используется для проверки, что аргумент был предоставлен
        )

        # Проверяем, что метод train был вызван с правильными параметрами
        mock_trainer_instance.train.assert_called_with(dataloaders["train"], dataloaders["valid"], epochs=(0, max_epoch))

        # Проверяем, что возвращаемое значение соответствует ожидаемому
        self.assertEqual(history, "mock_history")


# Mock-реализация модели
class SimpleTestModel(nn.Module):
    def __init__(self):
        super(SimpleTestModel, self).__init__()
        # Создаем один параметр модели для тестирования
        self.param = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        return x * self.param


# Mock-реализация функции потерь
class MockLossFunction:
    def __call__(self, outputs, targets):
        mock_loss = Mock()
        mock_loss.item.return_value = 0.5  # Имитация возвращаемого значения функции потерь
        return mock_loss


# Mock-реализация DataLoader
class MockDataLoader:
    def __iter__(self):
        # Возвращаем итератор, который имитирует пары (входные данные, метки)
        # Здесь мы просто используем примеры с мок-данными
        return iter([(
            torch.randn(1, 10),  # Пример входных данных
            torch.tensor([1])    # Пример меток
        ) for _ in range(10)])  # 10 примеров в DataLoader


if __name__ == '__main__':
    unittest.main()
