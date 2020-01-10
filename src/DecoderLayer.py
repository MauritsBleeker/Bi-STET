import torch.nn as nn
from SublayerConnection import SublayerConnection
from utils.utils_functions import clones


class DecoderLayer(nn.Module):
	"""
	Decoder is made of self-attn, src-attn, and feed forward (defined below)
	"""

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		"""
		
		:param size:
		:param self_attn:
		:param src_attn:
		:param feed_forward:
		:param dropout:
		"""
		
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"""
		Forward pass of the decoder layer
		:param x:
		:param memory:
		:param src_mask:
		:param tgt_mask:
		:return:
		"""
		
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)
