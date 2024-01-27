import tempfile
import shutil
from time import sleep

import torch
from torch.utils.data import DataLoader, TensorDataset

from trainer import ModelTrainer
from storage import TrainingProgressStorage
from metrics import MetricsLogger

def accuracy(outputs, labels):
    with torch.no_grad():
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total


storage_dir = tempfile.mkdtemp()
shutil.rmtree(storage_dir)
storage = TrainingProgressStorage(storage_dir)
print(storage_dir)

model = torch.nn.Linear(10, 2)  # Предполагаем, что у нас 2 класса
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

metrics_logger = MetricsLogger()
metrics_logger.add_metric_function("accuracy", accuracy)

# Создаем объект ModelTrainer
trainer = ModelTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=torch.device('cpu'),
    save_every_k_epochs=1,
    storage=storage,
    metrics_logger=metrics_logger
)


def print_callbacks(stage, history, **kwargs):
    if stage == 'start_epoch':
        pass
    #     epoch = kwargs.get('current_epoch')
    #     print(f"Начало эпохи {epoch}")
    elif stage == 'end_epoch':
        print()
    elif stage == 'end_batch':
        epoch_hist = history[-1]
        epoch = epoch_hist["epoch"]
        train_inf, valid_inf, test_inf = "", "", ""

        h = epoch_hist["training"]
        if len(h):
            batches = len(h["batches"])
            loss = h["metrics"]["loss"]
            acc = h["metrics"]["accuracy"]
            train_inf = f"TRAIN ({batches}): loss= {loss:.4f}" + (f", acc: {acc:.2f}" if acc is not None else "")

        h = epoch_hist["validation"]
        if len(h):
            batches = len(h["batches"])
            loss = h["metrics"]["loss"]
            acc = h["metrics"]["accuracy"]
            valid_inf = f"\tVALID ({batches}): loss= {loss:.4f}" + (f", acc: {acc:.2f}" if acc is not None else "")

        h = epoch_hist["testing"]
        if len(h):
            batches = len(h["batches"])
            loss = h["metrics"]["loss"]
            acc = h["metrics"]["accuracy"]
            test_inf = f"\tTEST ({batches}): loss= {loss:.4f}" + (f", acc: {acc:.2f}" if acc is not None else "")

        # Используем carriage return (\r) для перезаписи строки
        print(f"\rEPOCH ({epoch}) {train_inf}{valid_inf}{test_inf}", end="")
        sleep(0.2)


# Добавление callback-функций в ModelTrainer
trainer.callbacks.append(print_callbacks)

inputs = torch.randn(10, 10)
targets = torch.randint(0, 2, (10,))  # Генерация целочисленных меток для 2 классов
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=2)

# Запуск обучения
trainer.train(data_loader, data_loader, epochs=3)

best_epoch = trainer.restore(metric_path='metrics.validation.loss', fn='min')
print(f"Восстановлена лучшая эпоха {best_epoch}")
trainer.train(data_loader, data_loader, epochs=5)

epoch_4 = trainer.restore(metric_path='epoch', fn='eq', value=4)
print(f"Восстановлена эпоха {epoch_4}")
trainer.train(data_loader, data_loader, epochs=5)

