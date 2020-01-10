import torch.nn as nn
from LayerNorm import LayerNorm
from utils.utils_functions import clones


class Decoder(nn.Module):
	"""
	Generic N layer decoder with masking.
	"""

	def __init__(self, layer, N):
		"""
		
		:param layer: layer class
		:param N: Number of Decoder layers as stack
		"""
		
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""
		Forward pass for the decoder
		:param x: data
		:param memory: embeddings from the encoder
		:param src_mask: mask for the source
		:param tgt_mask: target mask, hide the targets for the next time step predicting during parallel training of the decoder
		:return: decoder output
		"""
		
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)
