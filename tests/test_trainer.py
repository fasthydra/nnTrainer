import copy
import torch
import torch.utils.data as data
from nn_trainer import ModelTrainer
from unittest.mock import Mock


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


def test_training_process():
    # Использование реальной модели, оптимизатора, планировщика и критерия
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.BCELoss()

    # Создание тестовых данных и DataLoader
    test_inputs = torch.randn(10, 2)
    test_targets = torch.randint(0, 2, (10, 1)).float()  # Тензор меток для бинарной классификации
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

    # Имитация вызова метода fit
    trainer.fit(test_data_loader, 2, mode='training')

    # Проверки вызовов и состояния после обучения
    assert len(trainer.history) == 2, "История обучения должна содержать записи за каждую эпоху"
    assert model.state_dict() != initial_model_state, "Состояние модели должно измениться после обучения"
    assert trainer.history[0]['epoch'] == 1, "Первая запись в истории должна соответствовать первой эпохе"
    assert trainer.history[1]['epoch'] == 2, "Вторая запись в истории должна соответствовать второй эпохе"
    assert 'training' in trainer.history[0], "Должны быть сохранены метрики обучения для первой эпохи"
    assert 'training' in trainer.history[1], "Должны быть сохранены метрики обучения для второй эпохи"








