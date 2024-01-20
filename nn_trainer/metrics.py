import time
import torch
from typing import Callable, Dict, List, Optional, Union, Any


class MetricsLogger:
    def __init__(self, history: Optional[List[Dict[str, Any]]] = None):
        """
        Инициализация MetricsLogger.
        """
        self.history = history if history is not None else []
        self._epoch_history = None
        self._mode_history = None
        self._batch_history = None

        self.batch_metric_functions: Dict[str, Callable] = {}
        self.total_metric_functions: Dict[str, Callable] = {}  # ToDo: переименовать

    # ToDo: переделать на инициализацию
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

    def calculate_mode_metrics(self, data_loader: torch.utils.data.DataLoader):
        """
        Вычисляет метрики эпохи.
        :param data_loader: DataLoader для текущей эпохи.
        """
        for name, func in self.total_metric_functions.items():
            mode_metric_value = func(data_loader)
            self._mode_history[name] = mode_metric_value

    def start_epoch(self, epoch: int):
        """
        Подготавливает logger к новой эпохе, сбрасывая необходимые данные.
        """
        epoch_start_time = time.time()

        self.history.append({'training': {}, 'validation': {}, 'testing': {}})
        self._epoch_history = self.history[-1]
        self._epoch_history["epoch"] = epoch
        self._epoch_history["start_time"] = epoch_start_time

    def end_epoch(self):
        """
        Финализирует метрики за эпоху
        """
        epoch_duration = time.time() - self._epoch_history["start_time"]
        self._epoch_history["duration"] = epoch_duration

    def start_mode(self, mode: str):
        """
        Подготавливает logger к новой эпохе, сбрасывая необходимые данные.
        :param mode: Режим эпохи ('training', 'validation', 'testing').
        """
        mode_start_time = time.time()

        self._mode_history = self._epoch_history[mode]
        self._mode_history["start_time"] = mode_start_time
        self._mode_history["processed_data"] = 0
        self._mode_history["duration"] = None
        self._mode_history["batches"] = []
        self._mode_history["total"] = {name: 0.0 for name in self.total_metric_functions}
        self._mode_history["metrics"] = {}

    def end_mode(self):
        """
        Финализирует метрики за тот или иной режим эпохи, вычисляя средние значения и очищая временные данные.
        """
        mode_duration = time.time() - self._mode_history["start_time"]
        self._mode_history["duration"] = mode_duration

    def start_batch(self):
        """
        Фиксирует время начала обработки батча.
        """
        batch_start_time = time.time()

        batch_history = {
            "batch_number": len(self._mode_history["batches"]) + 1,
            "start_time": batch_start_time,
            "duration": None,
            "batch_size": None,
            "metrics": {}
        }

        self._mode_history["batches"].append(batch_history)
        self._batch_history = self._mode_history["batches"][-1]

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
        fix_time = time.time()

        # ToDo: может быть тут еще по эпохам и модам считать?
        self._batch_history["duration"] = fix_time - self._batch_history["start_time"]

        self._batch_history["batch_size"] = batch_size if batch_size else outputs.size(0)
        self._mode_history["processed_data"] += self._batch_history["batch_size"]

        if torch.is_tensor(loss) and loss.nelement() == 1:
            self._batch_history["metrics"]["loss"] = loss.item()
        elif isinstance(loss, float):
            self._batch_history["metrics"]["loss"] = loss
        else:
            raise ValueError("Loss must be a single-element tensor or a float")

        with torch.no_grad():

            for name, func in self.batch_metric_functions.items():
                try:
                    metric_value = func(outputs, labels)
                    if metric_value is not None:
                        self._batch_history["metrics"][name] = metric_value
                except Exception as e:
                    print(f"Ошибка при вычислении метрики {name}: {e}")

            for key, value in self._batch_history["metrics"].items():
                self._mode_history["total"][key] = self._mode_history["total"].get(key, 0) + \
                                                   value * self._batch_history["batch_size"]

            # ToDo: как-то это богомерзко!
            proc_data = 1 if self._mode_history["processed_data"] == 0 else self._mode_history["processed_data"]

            # Вычисление средних значений метрик за эпоху
            for key, total in self._mode_history["total"].items():
                self._mode_history["metrics"][key] = total / proc_data
