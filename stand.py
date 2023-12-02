import itertools
import torch
from torch.utils.data import DataLoader
from training import ModelTrainer, TrainingProgressStorage


class TrainStand:

	def __init__(self, datasets, max_epoch, optimizer, scheduler, optimizer_params, scheduler_params,
				 save_every_k_epochs=5, device=None, callbacks=None):
		self.datasets = datasets
		self.max_epoch = max_epoch
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.optimizer_params = optimizer_params
		self.scheduler_params = scheduler_params
		self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.callbacks = callbacks if callbacks else []
		self.save_every_k_epochs = save_every_k_epochs

	def launch(self, models, loss_functions, batch_sizes, learning_rates):
		combinations = itertools.product(
			models.keys(),
			loss_functions.keys(),
			learning_rates,
			batch_sizes
		)

		for model_name, loss_fn_name, start_lr, batch_size in combinations:
			model = models[model_name]
			loss_function = loss_functions[loss_fn_name]
			history = self.run(model_name, model,
							   loss_fn_name, loss_function,
							   start_lr, batch_size, self.max_epoch)

	def run(self, model_name, model, loss_fn_name, loss_fn, start_lr, batch_size, max_epoch):

		train_data, valid_data, test_data = self.get_dataloaders(self.datasets, batch_size)
		run_name = self.create_run_name(model_name, loss_fn_name, start_lr, batch_size)
		trainer = self.create_trainer(model, loss_fn, run_name)

		last_epoch = self.restore_trainer(run_name, trainer)

		if last_epoch < max_epoch:
			history = trainer.train(train_data, valid_data, epochs=(last_epoch, max_epoch))
		else:
			history = trainer.history
		return history

	def get_dataloaders(self, batch_size):
		train = DataLoader(self.datasets["train"], batch_size=batch_size, shuffle=True)
		valid = DataLoader(self.datasets["valid"], batch_size=batch_size, shuffle=True)
		test = DataLoader(self.datasets["test"], batch_size=batch_size, shuffle=True)
		return train, valid, test

	def create_run_name(self, model_name, loss_fn_name, start_lr, batch_size):
		return f"{model_name}_{loss_fn_name}_lr{start_lr}_bs{batch_size}"

	def create_trainer(self, model, loss_function, run_name):

		storage = TrainingProgressStorage(run_name)

		trainer = ModelTrainer(
			model=model,
			optimizer=self.optimizer,
			scheduler=self.scheduler,
			criterion=loss_function,
			device=self.device,
			callbacks=self.callbacks,
			history=[],
			storage=storage,
			save_every_k_epochs=5,
		)

		return trainer

	def restored_trainer(self, run_name, trainer):
		last_epoch = 1
		storage = TrainingProgressStorage(run_name)
		last_save = storage.get_saved('timestamp')
		if last_save:
			model_state, optimizer_state, scheduler_state, last_epoch, history = storage.restore(last_save['id'])
			trainer.model.load_state_dict(model_state)
			trainer.optimizer.load_state_dict(optimizer_state)
			trainer.scheduler.load_state_dict(scheduler_state)
		return last_epoch


