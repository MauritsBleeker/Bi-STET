import torch
import os
import torch.nn as nn


class LossCompute(nn.Module):
	
	def __init__(self, criterion, optimizer, opt=None, warmup=None, factor=1, model_size=0):
		"""

		:param criterion: loss function
		:param optimizer: optimization function
		"""
		
		super(LossCompute, self).__init__()
		self.criterion = criterion
		self.opt = opt
		self.optimizer = optimizer
		
		self._step = 1
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0
		
	def forward(self, x, y, device, tgt_mask=None):
		"""
		
		:param x: model output
		:param y: targets
		:param device: CPU or GPU
		:param tgt_mask: loss mask, to ignore the loss of the padding symbols
		:return: loss value
		"""
		
		loss = self.criterion(
			x.contiguous().view(-1, x.size(-1)),
			y.contiguous().view(-1,  y.size(-1)).float()
		)
		
		if tgt_mask is not None:
			tgt_mask = tgt_mask.to(tgt_mask)
			loss = tgt_mask.view(-1, tgt_mask.shape[2]).float() * loss
			loss = loss.sum() / x.shape[0]
			
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()
		else:
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		self._step += 1
		return loss
	
	def lr_rate_step(self, step=None):
		"""
		Learning rate according to the Attetnion is all you need paper
		:param step: current step
		:return:
		"""
		
		if step is None:
			step = self._step
		rate = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		return rate
	
	def load_optimizer(self, path, file, device):
		"""

		:param path: path where checkpoint is stored
		:param file: checkpoint file name
		:param device CPU or GPU
		:return:
		"""

		self.optimizer.load_state_dict((torch.load(os.path.join(path, file), map_location=device)))
