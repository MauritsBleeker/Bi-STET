import torch.nn as nn
import math


class Embeddings(nn.Module):
	"""
	Embedding class
	"""
	def __init__(self, d_model, vocab):
		"""
		
		:param d_model: dimensionality of the character embeddings
		:param vocab: number of character used for decoding
		"""
		
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		"""
		
		:param x: indices of the characters
		:return: embeddings
		"""
		
		return self.lut(x.long()) * math.sqrt(self.d_model)
