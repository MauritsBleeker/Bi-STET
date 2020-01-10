import logging
import torch
import torch.nn as nn
from Batch import Batch
from Optimizer import LossCompute
from utils.DataSummary import DataSummary
from utils.utils_functions import get_dataset_loader, concat_ltr_rtl
from torch.optim.lr_scheduler import MultiStepLR


class Trainer(object):
	"""
	Model Trainer Object
	"""
	def __init__(self, config, device, model):
		"""
		
		:param config: Config class
		:param device: CPU or GPU
		:param model: Bi-STET Class
		"""
		self.device = device
		self.config = config
		
		self.summaries = DataSummary(self.config.OUPUT_FOLDER, start_iteration=self.config.START_ITERATION)
		
		self.vision_transformer = model
		self.vision_transformer.model.to(self.device)
		
		logging.info("Model initialized")
		
		self.optimizer = torch.optim.Adadelta(self.vision_transformer.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=1e-5, rho=0.95)
		self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.config.LR_MILESTONES, gamma=self.config.LR_GAMMA)
		self.loss_compute = LossCompute(
			criterion=nn.KLDivLoss(size_average=self.config.SIZE_AVARAGE, reduce=self.config.REDUCE_LOSS),
			optimizer=self.optimizer,
			model_size=self.config.D_MODEL,
			warmup=self.config.WARMUP,
			factor=self.config.FACTOR,
		)

		self.datasets = {}

		logging.info("Loading datasets")

		self.dataset_iterators = self.get_datasets_iterators(self.config.TRAIN_DATASETS)
		self.start_iteration = self.config.START_ITERATION
		self.iteration = self.start_iteration
		
		if self.start_iteration > 1 and self.lr_scheduler is not None:
			self.forward_lr_scheduler()
			logging.info("Forwarded lr scheduler")
			
		logging.info("Datasets Loaded")
		
	def get_datasets_iterators(self, dataset_paths):
		"""
		
		:param dataset_paths: list with paths to the dataset
		:return:
		"""
		data_iterators = []
		for i, dataset_path in enumerate(dataset_paths):
			data_iterator = get_dataset_loader(
				root_folder=dataset_path[0],
				annotation_file=dataset_path[1],
				input_transform=self.config.INPUT_TRANSFORM,
				batch_size=int(self.config.SAMPLES_PER_BATCH / len(dataset_paths)),
				num_workers=self.config.NUM_WORKERS,
				load_pickle=self.config.LOAD_PICKLE,
				validation_set=False
			)
			self.datasets[i] = data_iterator
			data_iterators.append(enumerate(data_iterator))

		return data_iterators
	
	def get_batch(self):
		"""
		
		:return:
		"""

		batch_data = list()
		for i in range(len(self.dataset_iterators)):
			try:
				_, data = next(self.dataset_iterators[i])
			except StopIteration:
				self.dataset_iterators[i] = enumerate(self.datasets[i])
				_, data = next(self.dataset_iterators[i])
				
			batch_data.append(data)
			
		return Batch(batch_data, device=self.device, bidirectional=self.config.BIDIRECTIONAL_DECODING)
	
	def train(self):
		"""
		
		:return:
		"""
		self.vision_transformer.model.train()
		self.summaries.resert_summaries()
		logging.info("Start training from iterations: {}".format(self.config.START_ITERATION))
		
		for iteration in range(self.config.START_ITERATION, self.config.TRAIN_ITERATIONS):

			self.iteration = iteration
			try:
				if self.lr_scheduler is not None:
					self.lr_scheduler.step()
				else:
					self.loss_compute.lr_rate_step()

				batch = self.get_batch()
				ltr, rtl = self.vision_transformer.model.forward(
					batch.images,
					tgt_embedding_mask=batch.targets_embedding_mask,
					src_mask=None,
					ltr_targets=batch.ltr_targets,
					rtl_targets=batch.rtl_targets
				)

				out = concat_ltr_rtl(ltr, rtl)
				targets_y = concat_ltr_rtl(batch.ltr_targets_y, batch.rtl_targets_y)
				if self.config.BIDIRECTIONAL_DECODING:
					batch.target_mask = concat_ltr_rtl(batch.target_mask, batch.target_mask)

				loss_value = self.loss_compute(out, targets_y, device=self.device, tgt_mask=batch.target_mask)

				self.summaries.add_summaries(loss_value.cpu().detach().numpy())
			except Exception as e:
				self.summaries.error(e)
			
			if iteration % self.config.SUMMARY_INTERVAL == 0 and iteration > 1:
				self.summaries.give_summary(reset=True)
				
			if (iteration % self.config.STORE_ITERVAL == 0 and iteration > 1) or iteration == self.config.TRAIN_ITERATIONS - 1:
				self.vision_transformer.save_model(self.config.OUTFILE + '-' + str(iteration))
				self.vision_transformer.store_parameter_histogram(self.summaries)
	
	def forward_lr_scheduler(self):
		"""
		:return:
		"""
		for i in range(self.start_iteration):
			self.lr_scheduler.step()
			
	def store_training_checkpoint(self, path):
		"""
		
		:param path: path to load checkpoint
		:return:
		"""
		
		torch.save({
			'epoch': self.iteration,
			'model_state_dict': self.vision_transformer.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
		}, path)
