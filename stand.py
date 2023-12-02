import itertools
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from training import ModelTrainer, TrainingProgressStorage


class TrainStand:

	def __init__(self, datasets, max_epoch, optimizer_params, scheduler_params,
				 save_every_k_epochs=5, device=None, callbacks=None):
		self.datasets = datasets
		self.max_epoch = max_epoch
		self.optimizer_params = optimizer_params
		self.scheduler_params = scheduler_params
		self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.callbacks = callbacks if callbacks else []
		self.save_every_k_epochs = save_every_k_epochs

	def launch(self, models, loss_functions, batch_sizes, learning_rates):
		combinations = itertools.product(models.keys(), loss_functions.keys(), learning_rates, batch_sizes)
		for model_name, loss_fn_name, start_lr, batch_size in combinations:
			model = models[model_name]
			loss_function = loss_functions[loss_fn_name]
			dataloaders = self.get_dataloaders(self.datasets, batch_size)
			run_name = self.create_run_name(model_name, loss_fn_name, start_lr, batch_size)
			history = self.run(run_name, model, loss_function, dataloaders, start_lr, self.max_epoch)

	def run(self, run_name, model, loss_fn, data, start_lr, max_epoch):
		trainer = self.create_trainer(model, loss_fn)
		trainer.storage = TrainingProgressStorage(run_name)
		last_epoch = trainer.restore()
		if last_epoch < max_epoch:
			history = trainer.train(data["train"], data["valid"], epochs=(last_epoch, max_epoch))
		else:
			history = trainer.history
		return history

	def get_dataloaders(self, batch_size):
		return {
			"train": DataLoader(self.datasets["train"], batch_size=batch_size, shuffle=True),
			"valid": DataLoader(self.datasets["valid"], batch_size=batch_size, shuffle=True),
			"test": DataLoader(self.datasets["test"], batch_size=batch_size, shuffle=True)
		}


	def create_run_name(self, model_name, loss_fn_name, start_lr, batch_size):
		return f"{model_name}_{loss_fn_name}_lr{start_lr}_bs{batch_size}"

	def create_trainer(self, model, loss_function, start_lr):

		optimizer = self.init_optimizer(model, start_lr)
		scheduler = self.init_scheduler(optimizer)

		trainer = ModelTrainer(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			criterion=loss_function,
			device=self.device,
			callbacks=self.callbacks,
			save_every_k_epochs=5,
		)

		return trainer

	def init_optimizer(self, model, start_lr):
		optimizer_class = getattr(optim, self.optimizer_params["type"])
		optimizer_args = self.optimizer_params["args"].copy()
		optimizer_args["lr"] = start_lr  # Установка начального learning rate
		return optimizer_class(model.parameters(), **optimizer_args)

	def init_scheduler(self, optimizer):
		scheduler_class = getattr(optim.lr_scheduler, self.scheduler_params["type"])
		return scheduler_class(optimizer, **self.scheduler_params["args"])





