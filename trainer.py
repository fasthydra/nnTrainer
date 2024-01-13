# -*- coding: utf-8 -*-
import copy
import time
import torch
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from storage import TrainingProgressStorage
from metrics import MetricsLogger


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
        metrics_logger: Optional[MetricsLogger] = None,
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
        self.metrics_logger = metrics_logger if metrics_logger else MetricsLogger()
        self.history = copy.deepcopy(history) if history is not None else []
        self.callbacks = callbacks if callbacks else []

        self.save_progress = self._dummy_save_progress
        self.storage = storage
        self.save_every_k_epochs = save_every_k_epochs

        self.first_epoch = 1
        self._early_stop_best_val = float('inf')
        self._early_stop_epochs_without_improvement = 0

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
        if value is not None:
            self.save_progress = value.get_save_fn(self.model, self.optimizer, self.scheduler, self.history,
                                                   self.save_every_k_epochs)
        else:
            self.save_progress = self._dummy_save_progress

    @staticmethod
    def get_inputs(data):
        return data[0]

    @staticmethod
    def get_labels(data):
        return data[1]

    @staticmethod
    def get_outputs(model, inputs):
        return model(inputs)

    @staticmethod
    def get_loss(outputs, labels, loss_fn):
        return loss_fn(outputs, labels)

    def fit_eval_epoch(self, data_loader, mode='train') -> list:
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        for step, data in enumerate(data_loader):

            self.metrics_logger.start_batch()

            inputs = self.get_inputs(data)
            labels = self.get_labels(data)

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if mode == 'train':
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(mode == 'train'):
                outputs = self.get_outputs(self.model, inputs)
                loss = self.get_loss(self.criterion, outputs, labels)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            self.metrics_logger.end_batch(outputs, labels, loss.item())

            self._call_callbacks('end_batch', metrics=self.metrics_logger.batch_metrics)

    def fit(self, train_loader: torch.utils.data.DataLoader) -> None:
        self.fit_eval_epoch(train_loader, mode='train')
        self.metrics_logger.calculate_epoch_metrics(train_loader, "training")

    def eval(self, val_loader: torch.utils.data.DataLoader, mode: str = "validation") -> None:
        with torch.no_grad():
            self.fit_eval_epoch(val_loader, mode='eval')
            self.metrics_logger.calculate_epoch_metrics(val_loader, mode)

    def get_early_stop_value(self) -> float:
        return self.metrics_logger.metrics["validation"][-1]["loss"]

    def early_stop(self, patience) -> Union[None, bool]:

        if patience == -1:
            self._early_stop_best_val = float('inf')
            self._early_stop_epochs_without_improvement = 0
            return

        if not patience:
            return False

        early_stop_value = self.get_early_stop_value()

        if early_stop_value < self._early_stop_best_val:
            self._early_stop_best_val = early_stop_value
            self._early_stop_epochs_without_improvement = 0
        else:
            self._early_stop_epochs_without_improvement += 1

        if self._early_stop_epochs_without_improvement >= patience:
            return True

        return False

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: Union[int, Tuple[int, int]],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        patience: Optional[int] = None
    ) -> None:

        self._call_callbacks('start_train')

        self.early_stop(-1)

        start_epoch, end_epoch = self.first_epoch, epochs if isinstance(epochs, int) else epochs

        for epoch in range(start_epoch, end_epoch + 1):
            self._call_callbacks('start_epoch', epoch=epoch, end_epoch=end_epoch)

            self.fit(train_loader)
            self.eval(val_loader)

            if test_loader:
                self.eval(test_loader, mode="testing")

            self.save_progress(epoch)

            self._call_callbacks('end_epoch', epoch=epoch, end_epoch=end_epoch, epoch_metrics=epoch_metrics)

            if self.early_stop(patience):
                print(f"Ранняя остановка на эпохе {epoch}.")
                break

            if self.scheduler:
                self.scheduler.step()

        self._call_callbacks('end_train')

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
