import os
import copy
import torch
import logging
import torch.nn as nn
from Dataset import Dataset
from Decoder import Decoder
from DecoderLayer import DecoderLayer
from Embeddings import Embeddings
from Encoder import Encoder
from EncoderDecoder import EncoderDecoder
from EncoderLayer import EncoderLayer
from MultiHeadedAttention import MultiHeadedAttention
from PositionalEncoding import PositionalEncoding
from PositionwiseFeedForward import PositionwiseFeedForward
from PredictionLayer import PredictionLayer
from ResNet import ResNet, BasicBlock
from FeatureExtractionNetwork import FeatureExtractionNetwork


class BiSTET(object):
	
	def __init__(self, config, device):
		"""
		
		:param config: configuration class
		:param device: device (gpu/cpu) to store the model on

		"""

		self.device = device
		self.config = config
		
		self.model = self._make_model(
			num_tgt_chars=len(Dataset.CHAR_ID_MAP),
			N=self.config.N,
			h=self.config.H,
			d_model=self.config.D_MODEL,
			d_ff=self.config.D_FF,
			dropout=self.config.DROPOUT,
		)
	
	def __call__(self, *args, **kwargs):
		"""
		
		:param args:
		:param kwargs:
		:return:

		"""
		return self.model
	
	def _make_model(self, num_tgt_chars, N, d_model, d_ff, h, dropout):
		"""
		
		:param num_tgt_chars: output space
		:param N: number of decoder and encoder layers
		:param d_model: model dimensionality
		:param d_ff: hidden size of the feed-forward neural network
		:param h: number of attention heads
		:param dropout: dropout rate
		:return: model

		"""
		c = copy.deepcopy
		attn = MultiHeadedAttention(h, d_model)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		position = PositionalEncoding(d_model, dropout)

		if self.config.USE_RESNET:
			feature_extractor = ResNet(block=BasicBlock, layers=self.config.RESNET_LAYERS, d_model=self.config.D_MODEL)
		else:
			feature_extractor = FeatureExtractionNetwork(d_model=self.config.D_MODEL)

		direction_embed = Embeddings(d_model, 2)

		model = EncoderDecoder(
			encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
			decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
			tgt_embed=nn.Sequential(Embeddings(d_model, num_tgt_chars), c(position)),
			generator=PredictionLayer(d_model, num_tgt_chars),
			feature_extractor=feature_extractor,
			prediction_layer=PredictionLayer(d_model, len(Dataset.CHAR_ID_MAP)),
			bidirectional_decoding=self.config.BIDIRECTIONAL_DECODING,
			direction_embed=direction_embed,
			device=self.device
		)
		
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_normal_(p)
		
		logging.info("Model created")
		
		return model
	
	def save_model(self, file_name):
		"""
		
		:param file_name: file name to store the model
		:return:

		"""
		
		torch.save(self.model.state_dict(), os.path.join(self.config.OUPUT_FOLDER, file_name + '.cp'))
		logging.info("Saved model: {}".format(file_name))

	def load_model(self, path, file):
		"""
		
		:param path: file path where model is stored
		:param file: checkpoint file name
		:return:

		"""

		logging.info("Load model: {}, from memory".format(file))
		self.model.load_state_dict(torch.load(os.path.join(path, file), map_location=self.device))

	def store_parameter_histogram(self, summary_writer):
		"""
		
		:param summary_writer: summery writer class
		:return:

		"""
	
		for name, p in self.model.named_parameters():
			summary_writer.store_histogram(name, p)
