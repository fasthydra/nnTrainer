import pytest
import os
import json
from unittest.mock import Mock
import torch
import torch.nn as nn
import torch.optim as optim
from storage import TrainingProgressStorage

@pytest.fixture
def storage(tmp_path):
    return TrainingProgressStorage(directory=str(tmp_path))

@pytest.fixture
def dummy_model():
    return nn.Linear(10, 2)

@pytest.fixture
def dummy_optimizer(dummy_model):
    return optim.SGD(dummy_model.parameters(), lr=0.001)

@pytest.fixture
def dummy_scheduler(dummy_optimizer):
    return optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1, gamma=0.1)

def test_initialization(storage, tmp_path):
    assert tmp_path.exists(), "Директория для сохранения не создана"
    assert (tmp_path / 'index.json').is_file(), "Индексный файл не создан"

def test_update_index(storage, tmp_path):
    save_data = {"id": "test_save", "epoch": 1}
    storage.update_index(save_data)
    index_path = tmp_path / 'index.json'
    with open(index_path, 'r') as file:
        index_data = json.load(file)
    assert save_data in index_data['saves'], "Данные не были добавлены в индексный файл"

def test_save_and_restore(storage, dummy_model, dummy_optimizer, dummy_scheduler):
    # Здесь необходимо добавить код для создания dummy_model, dummy_optimizer, dummy_scheduler
    epoch = 1
    history = [{"accuracy": 0.8}]
    parameters = {"learning_rate": 0.001}
    metrics = {"loss": 0.1}
    storage.save(dummy_model, dummy_optimizer, dummy_scheduler, epoch, history, parameters, metrics)
    save_id = f'save_001'
    restored_data = storage.restore(save_id)
    assert restored_data, "Данные не были восстановлены"

def test_restore_nonexistent_save(storage):
    with pytest.raises(FileNotFoundError):
        storage.restore('nonexistent_save')


def test_get_save_fn_calls_save_correctly(storage, dummy_model, dummy_optimizer, dummy_scheduler):
    save_every_k_epochs = 2

    # Мокаем метод save
    storage.save = Mock()

    # Получаем функцию для сохранения прогресса
    save_progress = storage.get_save_fn(dummy_model, dummy_optimizer, dummy_scheduler, [], save_every_k_epochs)

    # Вызываем функцию save_progress для разных эпох и проверяем, вызывается ли save
    for epoch in range(1, 6):
        save_progress(epoch)
        if epoch % save_every_k_epochs == 0:
            # Проверяем вызов save с учетом дополнительных параметров
            storage.save.assert_called_with(dummy_model, dummy_optimizer, dummy_scheduler, epoch, [], {'learning_rate': 0.001}, {})
            storage.save.reset_mock()  # Сброс мока после утверждения
        else:
            storage.save.assert_not_called()
