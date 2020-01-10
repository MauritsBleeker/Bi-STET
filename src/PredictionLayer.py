import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
	
	"""
	Define standard linear feed forward + softmax for final prediction layer.
	"""
	
	def __init__(self, d_model, vocab):
		"""
		
		:param d_model: model dimensionality
		:param vocab: output vocabulary size
		"""
		
		super(PredictionLayer, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		"""
		
		:param x: input tensor
		:return: output distribution
		"""
		
		return F.log_softmax(self.proj(x), dim=-1)
