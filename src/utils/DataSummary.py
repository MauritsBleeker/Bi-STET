import logging
import numpy as np
from utils.Timer import Timer
from tensorboardX import SummaryWriter


class DataSummary(object):
	"""
	DataSummary class. Used to store all kind of summaries during ytaining
	"""
	def __init__(self, log_dir, start_iteration=0):
		"""
		
		:param log_dir: folder to store the summaries
		:param start_iteration: iteration where the training is starting
		"""
		
		self.loss_per_iteration = []
		self.iteration = start_iteration
		self.error_counter = 0
		self.timer = Timer()
		self.timer.tic()
		self.log_dir = log_dir
		self.writer = SummaryWriter(
			log_dir=self.log_dir
		)
	
	def add_summaries(self, loss_value, n_steps=1):
		"""
		add summaries to object
		:param loss_value: loss value of one forward pass or mean loss value
		:param n_steps: number of past iterations
		:return:
		"""
		self.loss_per_iteration.append(loss_value)
		self.iteration += n_steps
		
	def give_summary(self, reset=True):
		"""
		Store summaries and print them to std
		:param reset: reset summaries
		:return:
		"""
		mean_los = np.mean(self.loss_per_iteration)
		
		logging.info('Iteration: {}, Mean loss: {:.4f}, Mean run time: {:.4f}'.format(self.iteration, mean_los, self.timer.toc(average=False)))
		self.writer.add_scalar('training/mean_loss', mean_los, self.iteration)
		if reset:
			self.resert_summaries()
			self.timer.tic()
			
	def resert_summaries(self):
		"""
		Reset summaries in the model
		:return:
		"""
		self.loss_per_iteration = []
		
	def error(self, e):
		"""
		write error
		:return:
		"""
		self.error_counter += 1
		error_msg = 'Error in iteration: {}, Error: {}'.format(self.iteration, e)
		logging.info(error_msg)
		self.writer.add_text('Error', error_msg, self.iteration)
	
	def store_histogram(self, name, param):
		"""
		store model weight histograms
		:param name: layer name
		:param param: model param
		:return:
		"""
		
		self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.iteration)
