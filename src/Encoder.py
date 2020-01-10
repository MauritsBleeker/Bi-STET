import torch.nn as nn
from LayerNorm import LayerNorm
from utils.utils_functions import clones


class Encoder(nn.Module):
	"""
	Core encoder is a stack of N layers"
	"""
	
	def __init__(self, layer, N):
		"""
		
		:param layer: Layer class
		:param N: Number of encoder layers in the stack
		"""
		
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		"""
		
		:param x: data
		:param mask: mask, not used
		:return: forward pass
		"""
		
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)
