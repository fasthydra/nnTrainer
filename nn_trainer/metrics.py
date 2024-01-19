import time
import copy
import torch
from typing import Callable, Dict, List, Optional, Union, Any


class MetricsLogger:
    def __init__(self, history: Optional[List[Dict[str, Any]]] = None):
        """
        Инициализация MetricsLogger.
        """
        self.history = history if history is not None else []
        self.epoch_metrics: Dict[str, Any] = {'training': {}, 'validation': {}, 'testing': {}}
        self.mode_metrics: Dict[str, Any] = {}
        self.batches_metrics: List[Dict] = []
        self.total_metrics: Dict[str, float] = {}
        
        self.batch_metric_functions: Dict[str, Callable] = {}
        self.total_metric_functions: Dict[str, Callable] = {}

        self._current_epoch = None
        
        self._processed_data: int = 0
        self._batch_sizes: List[int] = []
        self._batch_number: int = 0

        self._epoch_start_time: float = 0.0
        self._mode_start_time: float = 0.0
        self._batch_start_time: float = 0.0

    def add_metric_function(self, name: str, func: Callable, metric_type: str = "batch"):
        """
        Добавляет функцию метрики в зависимости от типа (батч или эпоха).

        :param name: Название метрики.
        :param func: Функция вычисления метрики.
        :param metric_type: Тип метрики ('batch' или 'epoch').
        """
        if metric_type not in ["batch", "epoch"]:
            raise ValueError(f"Недопустимый тип метрики: {metric_type}")

        if metric_type == "batch":
            self.batch_metric_functions[name] = func
        elif metric_type == "epoch":
            self.total_metric_functions[name] = func

        self.total_metrics[name] = 0.0

    def calculate_mode_metrics(self, data_loader: torch.utils.data.DataLoader):
        """
        Вычисляет метрики эпохи.

        :param data_loader: DataLoader для текущей эпохи.
        """
        mode_metrics = {}
        for name, func in self.total_metric_functions.items():
            mode_metric_value = func(data_loader)
            mode_metrics[name] = mode_metric_value

        self.mode_metrics["metrics"].update(mode_metrics)

    def start_epoch(self, epoch: int):
        """
        Подготавливает logger к новой эпохе, сбрасывая необходимые данные.
        """
        self._epoch_start_time = time.time()
        self._current_epoch = epoch
        self.epoch_metrics = {"epoch": self._current_epoch, 'training': {}, 'validation': {}, 'testing': {}}

    def start_mode(self, epoch: int):
        """
        Подготавливает logger к новой эпохе, сбрасывая необходимые данные.
        """
        self._mode_start_time = time.time()
        self._processed_data = 0
        self._batch_sizes = []

        self.batches_metrics = []
        self.total_metrics = {name: 0.0 for name in self.total_metric_functions}
        self.mode_metrics = {}

    def start_batch(self):
        """
        Фиксирует время начала обработки батча.
        """
        self._batch_start_time = time.time()
        self._batch_number = 0

    def end_batch(self, outputs: torch.Tensor, labels: torch.Tensor,
                  loss: Union[torch.Tensor, float],
                  batch_size: Optional[int] = None):
        """
        Обновляет метрики после обработки батча и фиксирует его продолжительность.
        Включает в себя логгирование потерь (loss).

        :param outputs: Выходные данные модели для батча (тензор).
        :param labels: Истинные метки для батча (тензор).
        :param batch_size: Размер батча. Если None, размер батча вычисляется из outputs.
        :param loss: Значение потерь для батча, может быть тензором или float.
        """

        if batch_size is None:
            batch_size = outputs.size(0)

        self._processed_data += batch_size

        batch_metrics = {
            "batch_number": self._batch_number,
            "duration": time.time() - self._batch_start_time,
            "batch_size": batch_size,
            "processed_data": self._processed_data,
            "metrics": {}
        }

        if torch.is_tensor(loss) and loss.nelement() == 1:
            batch_metrics["metrics"]["loss"] = loss.item()
        elif isinstance(loss, float):
            batch_metrics["metrics"]["loss"] = loss
        else:
            raise ValueError("Loss must be a single-element tensor or a float")

        with torch.no_grad():

            for name, func in self.batch_metric_functions.items():
                try:
                    metric_value = func(outputs, labels)
                    if metric_value is not None:
                        batch_metrics["metrics"][name] = metric_value
                except Exception as e:
                    print(f"Ошибка при вычислении метрики {name}: {e}")

            for key, value in batch_metrics["metrics"].items():
                self.total_metrics[key] = self.total_metrics.get(key, 0) + value * batch_size

            self.batches_metrics.append(batch_metrics)

    def end_mode(self, mode: str):
        """
        Финализирует метрики за тот или иной режим эпохи, вычисляя средние значения и очищая временные данные.

        :param mode: Режим эпохи ('training', 'validation', 'testing').
        """
        mode_metrics = {
            "duration": time.time() - self._mode_start_time,
            "batches": copy.deepcopy(self.batches_metrics),
            "total": copy.deepcopy(self.total_metrics),
            "metrics": {}
        }

        proc_data = 1 if self._processed_data == 0 else self._processed_data
        
        # Вычисление средних значений метрик за эпоху
        for key, total in mode_metrics["total"].items():
            mode_metrics["metrics"][key] = total / proc_data

        self.mode_metrics = mode_metrics
        self.epoch_metrics[mode] = mode_metrics

    def end_epoch(self):
        """
        Финализирует метрики за эпоху
        """
        self.epoch_metrics["duration"] = time.time() - self._epoch_start_time
        self.history.append(self.epoch_metrics)
