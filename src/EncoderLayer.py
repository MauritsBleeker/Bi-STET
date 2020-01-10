import torch.nn as nn
from SublayerConnection import SublayerConnection
from utils.utils_functions import clones


class EncoderLayer(nn.Module):
	"""
	Encoder is made up of self-attn and feed forward (defined below)
	"""
	def __init__(self, size, self_attn, feed_forward, dropout):
		"""
		
		:param size: size of the output
		:param self_attn: self attention function
		:param feed_forward: feed forward layer
		:param dropout: dropout rate
		"""
		
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"""
		
		:param x: input tensor
		:param mask: mask, not used
		:return:
		"""
		
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)
