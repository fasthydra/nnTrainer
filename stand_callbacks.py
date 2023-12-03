import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from stand import TrainStand

# Определение небольших тестовых моделей
models = {
    "simple_model": nn.Linear(10, 2)
}

# Определение функций потерь
loss_functions = {
    "cross_entropy": nn.CrossEntropyLoss()
}

# Создание тестовых датасетов
x_train, y_train = torch.randn(100, 10), torch.randint(0, 2, (100,))
x_valid, y_valid = torch.randn(100, 10), torch.randint(0, 2, (100,))
x_test, y_test = torch.randn(100, 10), torch.randint(0, 2, (100,))

datasets = {
    "train": TensorDataset(x_train, y_train),
    "valid": TensorDataset(x_valid, y_valid),
    "test": TensorDataset(x_test, y_test)
}

# Параметры для оптимизатора и планировщика
optimizer_params = {
    "type": "SGD",
    "args": {"momentum": 0.9}
}

scheduler_params = {
    "type": "StepLR",
    "args": {"step_size": 30, "gamma": 0.1}
}


# Функции для callback
def example_callback(stage, context, **kwargs):
    match stage:
        case 'before_run':
            print(f"Before run: model={kwargs['model']}, loss_fn={kwargs['loss_fn']}, lrate={kwargs['lrate']}, batch_size={kwargs['batch_size']}")
        case 'after_run':
            print(f"After run: model={kwargs['model']}, loss_fn={kwargs['loss_fn']}, lrate={kwargs['lrate']}, batch_size={kwargs['batch_size']}, history={kwargs['history']}")
        case 'start_run':
            print("Starting a new run...")
        case 'end_run':
            print("Run ended.")
        case _:
            print(f"Unknown stage: {stage}. Context: {context}, Arguments: {kwargs}")



# Инициализация класса TrainStand
train_stand = TrainStand(
    datasets=datasets,
    max_epoch=10,
    optimizer_params=optimizer_params,
    scheduler_params=scheduler_params,
    callbacks=[example_callback]
)

# Запуск процесса обучения
train_stand.launch(
    models=models,
    loss_functions=loss_functions,
    batch_sizes=[16],
    learning_rates=[0.01]
)

