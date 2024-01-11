import torch
from torch.utils.data import DataLoader, TensorDataset
from trainer import ModelTrainer


class CustomModelTrainer(ModelTrainer):
    def get_predictions_from_model_output(self, output):
        # Предполагаем, что output - это словарь с ключами 'y_pred' и 'latent_code'
        return output['y_pred']


def simple_callback(stage, trainer, **kwargs):
    if stage == 'model_outputs':
        print("Model outputs processed in callback")
        # Здесь может быть дополнительная логика обработки выходных данных модели


# Фиктивная модель, которая возвращает словарь с двумя ключами
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)  # Пример обучаемого слоя

    def forward(self, x):
        output = self.linear(x)
        return {'y_pred': output, 'latent_code': output}


# Фиктивные данные и загрузчик данных
x = torch.randn(10, 3)  # 10 примеров, каждый размером 3
y = torch.randn(10, 1)  # 10 меток
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset, batch_size=2)

# Инициализация компонентов для тренера
model = DummyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Создание экземпляра CustomModelTrainer
trainer = CustomModelTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=None,
    criterion=criterion,
    callbacks=[simple_callback]
)

# Тестирование
trainer.fit(data_loader)
