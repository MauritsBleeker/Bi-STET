import torch.nn as nn
import torch
import numpy as np
from utils.utils_functions import get_attention_distributions


class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many
	other models.
	"""

	def __init__(self, encoder, decoder, tgt_embed, generator, prediction_layer, feature_extractor, device, direction_embed=None, bidirectional_decoding=False):

		"""

		:param encoder: Encoder class
		:param decoder: Decoder class
		:param tgt_embed: lookup with the target embeddings
		:param prediction_layer: Ouput layer
		:param feature_extractor: CNN for feature extraction
		:param device: CPU or GPU
		:param direction_embed: Lookup for the direction embeddings
		:param bidirectional_decoding: If true, decode bidirectional
		"""
		
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.tgt_embed = tgt_embed
		self.generator = generator
		self.feature_extractor = feature_extractor
		self.prediction_layer = prediction_layer
		self.direction_embed = direction_embed
		self.bidirectional_decoding = bidirectional_decoding
		self.device = device

		self.ltr_attn_dist = None
		self.rtl_attn_dist = None

	def forward(self, image, src_mask, tgt_embedding_mask, ltr_targets, rtl_targets=None):
		"""
		Take in and process masked src and target sequences.
		:param image: input image
		:param src_mask: mask for encoder input, not used
		:param tgt_embedding_mask: mask for the target embeddings
		:param ltr_targets: targets left-to-right decoding
		:param rtl_targets: targets right-to-left decoding
		:return:
		"""
		
		return self.decode(self.encode(image, src_mask), src_mask, tgt_embedding_mask, ltr_targets, rtl_targets)

	def encode(self, image, src_mask):
		"""
		
		:param image: input image
		:param src_mask: mask for the input image, not used
		:return: encoded image representation
		"""

		return self.encoder(self.feature_extractor(image), src_mask)

	def decode(self, memory, src_mask, ltr_tgt_mask, ltr_targets, rtl_targets=None, rtl_tgt_mask=None):
		"""
		
		:param memory: the encoded image embeddings
		:param src_mask:
		:param ltr_tgt_mask: masking out the future targets
		:param ltr_targets: output targets
		:param rtl_targets: targets of the right-to-left sequence
		:param rtl_tgt_mask: targets of the left-to-right sequence
		:return:
		"""
		nbatches = memory.size(0)

		if not self.bidirectional_decoding or rtl_targets is None:
			ltr = self.prediction_layer(self.decoder(self.tgt_embed(ltr_targets) + self.direction_embed(torch.from_numpy(np.zeros((nbatches, 1))).to(self.device)), memory, src_mask, ltr_tgt_mask)), None
			self.ltr_attn_dist = get_attention_distributions(self.encoder, self.decoder)
			return ltr
		else:
			if rtl_tgt_mask is None:
				rtl_tgt_mask = ltr_tgt_mask

			ltr = self.prediction_layer(self.decoder(self.tgt_embed(ltr_targets) + self.direction_embed(torch.from_numpy(np.zeros((nbatches, 1))).to(self.device)), memory, src_mask, ltr_tgt_mask))
			self.ltr_attn_dist = get_attention_distributions(self.encoder, self.decoder)
			rtl = self.prediction_layer(self.decoder(self.tgt_embed(rtl_targets) + self.direction_embed(torch.from_numpy(np.ones((nbatches, 1))).to(self.device)), memory, src_mask, rtl_tgt_mask))
			self.rtl_attn_dist = get_attention_distributions(self.encoder, self.decoder)

			return ltr, rtl
