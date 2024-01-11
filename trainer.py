# -*- coding: utf-8 -*-
import copy
import time
import torch
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from storage import TrainingProgressStorage


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
        return dict()

    def get_model_inputs(self, inputs, labels):
        return inputs

    def get_labels(self, inputs, labels):
        return labels

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
                model_inputs = self.get_model_inputs(inputs, labels)
                outputs = self.model(model_inputs)
                self._call_callbacks('model_outputs', model_outputs=outputs)

                y_true = self.get_labels(inputs, labels)
                y_pred = self.get_predictions_from_model_output(outputs)
                loss_args = self.get_loss_args_from_model_output(outputs)

                loss = self.criterion(y_pred, y_true, **loss_args)

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
        epochs: Union[int, Tuple[int, int]],
        test_loader: Optional[torch.utils.data.DataLoader] = None,
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

            epoch_metrics = {}

            train_loss = self.fit(train_loader)
            train_score = self.score_function(self.model, train_loader)

            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            epoch_metrics['train_loss'] = train_loss
            epoch_metrics['train_score'] = train_score

            val_loss = self.eval(val_loader)
            val_score = self.score_function(self.model, val_loader)
            epoch_metrics['val_loss'] = val_loss
            epoch_metrics['val_score'] = val_score

            if test_loader:
                test_loss = self.eval(test_loader)
                test_score = self.score_function(self.model, test_loader)
                epoch_metrics['test_loss'] = test_loss,
                epoch_metrics['test_score'] = test_score

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
