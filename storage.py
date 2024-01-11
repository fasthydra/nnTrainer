# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import json
import torch
from typing import Any, Dict, Callable, Optional


class TrainingProgressStorage:
    def __init__(self, directory: str):
        """
        Инициализирует хранилище для управления прогрессом обучения.

        :param directory: Путь к директории для сохранения и восстановления данных.
        """
        self.directory = Path(directory)
        self.index_file_path = self.directory / 'index.json'
        self.initialize_index()
        self.next_save_id = self._read_index()['next_save_id']

    def initialize_index(self):
        """Инициализирует индексный файл, если он не существует."""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        if not self.index_file_path.is_file():
            self._write_index({'next_save_id': 1, 'saves': []})

    def update_index(self, save_data: Dict[str, Any]):
        """
        Обновляет индексный файл с новыми данными о сохранении.

        :param save_data: Данные для добавления в индексный файл.
        """
        index_data = self._read_index()
        index_data['saves'].append(save_data)
        index_data['next_save_id'] = self.next_save_id + 1
        self._write_index(index_data)

    def get_save_fn(self, model, optimizer, scheduler, history, save_every_k_epochs: int) -> Callable:
        """
        Возвращает функцию для сохранения прогресса обучения.

        :param model: Обучаемая модель.
        :param optimizer: Оптимизатор.
        :param scheduler: Планировщик скорости обучения.
        :param history: История обучения.
        :param save_every_k_epochs: Частота сохранения (каждые k эпох).
        :return: Функция сохранения прогресса.
        """
        def save_progress(epoch: int):
            if epoch % save_every_k_epochs == 0:
                parameters = {'learning_rate': optimizer.param_groups[0]['lr']}
                metrics = history[-1] if history else {}
                self.save(model, optimizer, scheduler, epoch, history, parameters, metrics)

        return save_progress

    def save(self,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
             epoch: int,
             history: list,
             parameters: Dict[str, Any],
             metrics: Dict[str, Any]):
        """
        Сохраняет текущее состояние обучения.

        :param model: Модель для сохранения.
        :param optimizer: Оптимизатор для сохранения.
        :param scheduler: Планировщик для сохранения.
        :param epoch: Текущая эпоха обучения.
        :param history: История обучения.
        :param parameters: Параметры обучения.
        :param metrics: Метрики обучения.
        """
        save_folder = self._generate_save_folder()
        torch.save(model.state_dict(), save_folder / 'model.pth')
        torch.save(optimizer.state_dict(), save_folder / 'optimizer.pth')
        torch.save(scheduler.state_dict(), save_folder / 'scheduler.pth')
        history_file = save_folder / 'history.json'
        with history_file.open('w') as f:
            json.dump(history, f, indent=4)

        save_data = {
            'id': f'save_{str(self.next_save_id).zfill(3)}',
            'epoch': epoch,
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'path': str(save_folder),
            'metrics': metrics
        }
        self.update_index(save_data)
        self.next_save_id += 1

    def restore(self, save_id: str):
        """
        Восстанавливает состояние обучения из указанного сохранения.

        :param save_id: Идентификатор сохранения для восстановления.
        :return: Состояние модели, оптимизатора, планировщика и истории обучения.
        """
        index_data = self._read_index()
        save_data = next((item for item in index_data['saves'] if item['id'] == save_id), None)
        if save_data:
            save_folder = Path(save_data['path'])
            model_state = torch.load(save_folder / 'model.pth')
            optimizer_state = torch.load(save_folder / 'optimizer.pth')
            scheduler_state = torch.load(save_folder / 'scheduler.pth')
            with (save_folder / 'history.json').open('r') as f:
                history = json.load(f)
            saved_progress = {
                "model": model_state,
                "optimizer": optimizer_state,
                "scheduler": scheduler_state,
                "epoch": save_data['epoch'],
                "history": history
            }
            return saved_progress
        else:
            raise FileNotFoundError(f"No save found with id '{save_id}'")

    def get_saved(self, metric_path: str, fn: str = "max"):
        """
        Возвращает точку сохранения на основе указанного метрического параметра.

        :param metric_path: Путь к метрическому параметру (например, 'loss', 'metrics.accuracy').
        :param fn: Функция для сравнения ('max' или 'min').
        :return: Идентификатор сохранения и соответствующие данные.
        """
        index_data = self._read_index()

        saves = index_data.get('saves')

        if not saves:
            return {}

        def extract_metric(save, path):
            keys = path.split('.')
            for key in keys:
                save = save.get(key, {})
            if isinstance(save, str):
                try:
                    return datetime.fromisoformat(save)
                except ValueError:
                    return save
            return save if isinstance(save, (float, int)) else float('-inf' if fn == 'max' else 'inf')

        if fn == "max":
            save_data = max(saves, key=lambda x: extract_metric(x, metric_path))
        else:  # min
            save_data = min(saves, key=lambda x: extract_metric(x, metric_path))

        return save_data

    def _generate_save_folder(self) -> Path:
        """
        Генерирует путь к папке для сохранения.

        :return: Путь к папке для сохранения.
        """
        save_folder = self.directory / f'save_{str(self.next_save_id).zfill(3)}'
        if not save_folder.exists():
            save_folder.mkdir(parents=True, exist_ok=True)
        return save_folder

    def _read_index(self) -> Dict[str, Any]:
        """
        Читает данные из индексного файла.

        :return: Данные из индексного файла.
        """
        with self.index_file_path.open('r') as file:
            return json.load(file)

    def _write_index(self, data: Dict[str, Any]):
        """
        Записывает данные в индексный файл.

        :param data: Данные для записи.
        """
        with self.index_file_path.open('w') as file:
            json.dump(data, file, indent=4)
