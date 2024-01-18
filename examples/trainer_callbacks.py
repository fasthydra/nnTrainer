import tempfile
import shutil

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

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.MSELoss()

metrics_logger = MetricsLogger()
metrics_logger.add_metric_function("accuracy", accuracy)

# Создаем объект ModelTrainer
trainer = ModelTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=torch.device('cpu'),
    save_every_k_epochs=5,
    storage=storage,
    metrics_logger=metrics_logger
)


def trainer_callbacks(stage, metrics, **kwargs):
    if stage == 'start_epoch':
        epoch = kwargs.get('current_epoch')
        print(f"Начало эпохи {epoch}")
    elif stage == 'end_epoch':
        epoch = kwargs.get('current_epoch')
        training_metrics = metrics['epoch']['training']
        validation_metrics = metrics['epoch']['validation']

        train_loss = training_metrics.get('loss')
        val_loss = validation_metrics.get('loss')
        train_accuracy = training_metrics.get('accuracy')
        val_accuracy = validation_metrics.get('accuracy')

        print(f"Эпоха {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}, Val Acc: {val_accuracy:.2f}")
    elif stage == 'end_batch':
        # Получение информации о последнем батче
        batch_metrics = metrics['batches'][-1]

        loss = batch_metrics["metrics"].get('loss')
        accuracy = batch_metrics["metrics"].get('accuracy', None)  # Может не быть accuracy
        batch_number = len(metrics['batches'])  # Порядковый номер текущего батча

        # Используем carriage return (\r) для перезаписи строки
        print(f"\rОбработка батча {batch_number}: Loss: {loss:.4f}" +
              (f", Acc: {accuracy:.2f}" if accuracy is not None else ""), end="")


# Добавление callback-функций в ModelTrainer
trainer.callbacks.append(trainer_callbacks)

# Создание DataLoader с некорректными данными
inputs = torch.randn(10, 10)
targets = torch.randn(10, 2)
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=2)

# Запуск обучения
trainer.train(data_loader, data_loader, epochs=(0, 5))
