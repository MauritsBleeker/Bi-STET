import torch.nn as nn
from LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		"""
		
		:param size: embedding size for layer norm
		:param dropout: dropout rate s
		"""
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"""
		Apply residual connection to any sublayer with the same size.
		:param x:
		:param sublayer:
		:return:
		"""
		return x + self.dropout(sublayer(self.norm(x)))
