import time
import torch
from typing import Callable, Dict, List, Optional


class MetricsLogger:
    def __init__(self):
        """
        Инициализация MetricsLogger.
        """
        self.metrics: Dict[str, List[Dict]] = {"training": [], "validation": [], "testing": []}
        self.epoch_metrics: Dict[str, Dict[str, float]] = {"training": {}, "validation": {}, "testing": {}}
        self.batch_metrics: List[Dict] = []
        self.metric_functions: Dict[str, Callable] = {}
        self.epoch_metric_functions: Dict[str, Callable] = {}
        self.total_metrics: Dict[str, float] = {}
        self.processed_data: int = 0
        self.batch_sizes: List[int] = []
        self.epoch_start_time: float = 0.0
        self.batch_start_time: float = 0.0

    def add_metric_function(self, name: str, func: Callable, metric_type: str = "batch"):
        """
        Добавляет функцию метрики в зависимости от типа (батч или эпоха).

        :param name: Название метрики.
        :param func: Функция вычисления метрики.
        :param metric_type: Тип метрики ('batch' или 'epoch').
        """
        if metric_type == "batch":
            self.metric_functions[name] = func
        elif metric_type == "epoch":
            self.epoch_metric_functions[name] = func

        self.total_metrics[name] = 0.0

    def calculate_epoch_metrics(self, data_loader: torch.utils.data.DataLoader, mode: str):
        """
        Вычисляет метрики эпохи.

        :param data_loader: DataLoader для текущей эпохи.
        :param mode: Режим эпохи ('training', 'validation', 'testing').
        """
        epoch_metrics = {}
        for name, func in self.epoch_metric_functions.items():
            epoch_metric_value = func(data_loader)
            epoch_metrics[name] = epoch_metric_value

        self.epoch_metrics[mode] = epoch_metrics

    def start_epoch(self):
        """
        Подготавливает logger к новой эпохе, сбрасывая необходимые данные.
        """
        self.epoch_start_time = time.time()
        self.batch_metrics = []
        self.total_metrics = {name: 0.0 for name in self.metric_functions}
        self.processed_data = 0
        self.batch_sizes = []

    def start_batch(self):
        """
        Фиксирует время начала обработки батча.
        """
        self.batch_start_time = time.time()

    def end_batch(self, outputs: torch.Tensor, labels: torch.Tensor, loss_value: float,
                  batch_size: Optional[int] = None):
        """
        Обновляет метрики после обработки батча и фиксирует его продолжительность.
        Включает в себя логгирование потерь (loss).

        :param outputs: Выходные данные модели для батча (тензор).
        :param labels: Истинные метки для батча (тензор).
        :param batch_size: Размер батча. Если None, размер батча вычисляется из outputs.
        :param loss_value: Значение функции потерь для батча.
        """
        batch_duration = time.time() - self.batch_start_time

        if batch_size is None:
            batch_size = outputs.size(0)

        self.processed_data += batch_size

        batch_metrics = {
            "batch_duration": batch_duration,
            "metrics": {},
            "loss": loss_value,
            "batch_size": batch_size,
            "processed_data": self.processed_data,
        }

        with torch.no_grad():
            batch_metrics["metrics"]["loss"] = loss_value

            for name, func in self.metric_functions.items():
                metric_value = func(outputs, labels)
                batch_metrics["metrics"][name] = metric_value

            proc_data = 1 if self.processed_data == 0 else self.processed_data

            for key in batch_metrics["metrics"].keys():
                self.total_metrics[key] += batch_metrics["metrics"][key] * batch_size
                batch_metrics["metrics"][f"acc_{key}"] = self.total_metrics[key] / proc_data

            self.batch_metrics.append(batch_metrics)

    def end_epoch(self, mode: str):
        """
        Финализирует метрики за эпоху, вычисляя средние значения и очищая временные данные.

        :param mode: Режим эпохи ('training', 'validation', 'testing').
        """
        epoch_duration = time.time() - self.epoch_start_time
        epoch_metrics = {
            "epoch_duration": epoch_duration,
            "average_metrics": {},
            "total_metrics": {},
            "batch_metrics": self.batch_metrics
        }

        # Вычисление средних значений метрик за эпоху
        for key, total in self.total_metrics.items():
            if self.processed_data > 0:
                epoch_metrics["epoch_metrics"][key] = total / self.processed_data
            epoch_metrics["total_metrics"][key] = total

        # Добавление финализированных метрик в историю эпох
        self.metrics[mode].append(epoch_metrics)

        # Сброс временных данных для следующей эпохи
        self.reset()


''' 
ToDo:
Есть ошибка в проектировании уровней иерархии расчета метрик:
Сейчас (неправильно):
    epoch
        batch
Нужно (правильно):
    train
        epoch
            mode (train, valid, test)
                batch
'''
