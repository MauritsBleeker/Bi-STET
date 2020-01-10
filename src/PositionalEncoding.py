import torch
import math
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
	"""
	Implement the PE function.
	"""

	def __init__(self, d_model, dropout, max_len=5000):
		"""
		
		:param d_model: embeddings dimensionality
		:param dropout: dropout value
		:param max_len: max sequence length
		"""
		
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0., max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		
		:param x:
		:return:
		"""
		
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)
