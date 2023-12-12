# -*- coding: utf-8 -*-
import copy
import time
from pathlib import Path
from datetime import datetime
import json
import torch
from typing import Any, Dict, List, Tuple, Optional, Callable, Union


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

    def save(self, model, optimizer, scheduler, epoch: int, history: list, parameters: Dict[str, Any], metrics: Dict[str, Any]):
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


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: torch.nn.modules.loss._Loss,
        score_function=None,
        device: Optional[torch.device] = None,
        storage: Optional[TrainingProgressStorage] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        callbacks: Optional[List[Callable]] = None,
        save_every_k_epochs: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.score_function = score_function if score_function else self._dummy_score_function
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = copy.deepcopy(history) if history is not None else []
        self.callbacks = callbacks if callbacks else []

        self.save_progress = self._dummy_save_progress
        self.storage = storage
        self.save_every_k_epochs = save_every_k_epochs

        self.first_epoch = 1

        self.exec_context = {}  # данные контекста выполнения для использования в callbacks

    def _dummy_save_progress(self, epoch: int):
        """ A placeholder method for saving progress when no storage is provided. """
        pass

    def _dummy_score_function(self, model, data):
        """ A placeholder method for saving progress when no storage is provided. """
        return None

    def _call_callbacks(self, stage: str, **kwargs):
        self.exec_context[stage] = {'timestamp': time.time()}
        if self.callbacks:
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                for callback in self.callbacks:
                    callback(stage, self, **kwargs)
            if was_training:
                self.model.train()

    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, value):
        self._storage = value
        # Если storage не None, используем функцию сохранения от storage, иначе используем _dummy_save_progress
        if value is not None:
            self.save_progress = value.get_save_fn(self.model, self.optimizer, self.scheduler, self.history,
                                                   self.save_every_k_epochs)
        else:
            self.save_progress = self._dummy_save_progress

    def get_predictions_from_model_output(self, model_output):
        return model_output

    def get_loss_args_from_model_output(self, model_output):
        return None

    def fit_eval_epoch(self, data_loader, mode='train') -> float:
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        processed_data = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if mode == 'train':
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(mode == 'train'):
                outputs = self.model(inputs)
                y_pred = self.get_predictions_from_model_output(outputs)
                loss_args = self.get_loss_args_from_model_output(outputs)
                loss = self.criterion(y_pred, labels, **loss_args)
                self._call_callbacks('model_outputs', model_outputs=outputs)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            processed_data += inputs.size(0)

            self._call_callbacks('end_batch', loss=(running_loss / processed_data))

        loss = (running_loss / processed_data)

        return loss

    def fit(self, train_loader: torch.utils.data.DataLoader) -> float:
        return self.fit_eval_epoch(train_loader, mode='train')

    def eval(self, val_loader: torch.utils.data.DataLoader) -> float:
        with torch.no_grad():
            return self.fit_eval_epoch(val_loader, mode='eval')

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        epochs: Union[int, Tuple[int, int]],
        patience: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        self._call_callbacks('start_train')
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        if isinstance(epochs, int):
            start_epoch = self.first_epoch
            end_epoch = epochs
        else:
            start_epoch, end_epoch = epochs

        for epoch in range(start_epoch, end_epoch + 1):
            self._call_callbacks('start_epoch', epoch=epoch, end_epoch=end_epoch)

            train_loss = self.fit(train_loader)
            val_loss = self.eval(val_loader)
            test_loss = self.eval(test_loader)

            train_score = self.score_function(self.model, train_loader)
            val_score = self.score_function(self.model, val_loader)
            test_score = self.score_function(self.model, test_loader)

            epoch_metrics = {
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'train_score': train_score,
                'val_loss': val_loss,
                'val_score': val_score,
                'test_loss': test_loss,
                'test_score': test_score
            }
            self.history.append(epoch_metrics)

            self.save_progress(epoch)

            self._call_callbacks('end_epoch', epoch=epoch, end_epoch=end_epoch, epoch_metrics=epoch_metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch}.")
                break

            if self.scheduler:
                self.scheduler.step()

        self._call_callbacks('end_train')

        return self.history

    def restore(self):
        last_epoch = self.first_epoch
        last_save = self.storage.get_saved('timestamp') if self.storage else None
        if last_save:
            restored_progress = self.storage.restore(last_save['id'])
            self.model.load_state_dict(restored_progress["model"])
            self.optimizer.load_state_dict(restored_progress["optimizer"])
            self.scheduler.load_state_dict(restored_progress["scheduler"])
            self.history = restored_progress["history"]
            last_epoch = restored_progress["epoch"]
        return last_epoch
