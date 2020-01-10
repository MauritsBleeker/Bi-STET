import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
	"""
	Implements FFN equation.
	"""
	def __init__(self, d_model, d_ff, dropout=0.1):
		"""
		
		:param d_model: embeddings dimensionality
		:param d_ff: size hidden state feed forward neural net
		:param dropout: dropout rate
		"""
		
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		"""
		
		:param x: input tensor
		:return: output tensor
		"""
		return self.w_2(self.dropout(F.relu(self.w_1(x))))
