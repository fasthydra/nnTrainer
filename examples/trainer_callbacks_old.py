import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nn_trainer.trainer import ModelTrainer


class CustomModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Дополнительные инициализации (если необходимы)

    def get_predictions_from_output(self, output):
        # Реализация вашего собственного метода
        # Например, вы можете изменить вывод модели или применить некоторую постобработку
        # Здесь должен быть ваш код для переопределенного метода
        pass


def simple_callback(stage, trainer, **kwargs):
    if stage == 'start_epoch':
        print(f"Начало эпохи {kwargs['epoch']}")
    elif stage == 'end_epoch':
        metrics = kwargs['epoch_metrics']
        print(f"Эпоха {kwargs['epoch']} завершена. Train Loss: {metrics['train_loss']}, Val Loss: {metrics['val_loss']}")


# Создание синтетических данных для обучения и валидации
X_train, y_train = torch.randn(100, 10), torch.randint(0, 2, (100,))
X_val, y_val = torch.randn(50, 10), torch.randint(0, 2, (50,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

# Инициализация модели, оптимизатора, планировщика и функции потерь
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Инициализация ModelTrainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = ModelTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    callbacks=[simple_callback]
)

# Запуск обучения
history = trainer.train(train_loader, val_loader, val_loader, epochs=(0, 5))
