import torch
from math import log
import re
from Dataset import Dataset


class Word(object):
	"""
	class to decode a word during evaluation
	"""
	def __init__(self, device, beam_size=1):
		"""

		:param device: GPU or CPU
		:param beam_size: size of the beam, if decoding with beam search
		"""
		self.characters = ''
		self.probability = 1
		self.ended = False
		self.device = device
		self.targets = torch.tensor([[Dataset.GO_ID]] * beam_size).to(self.device)

	def greedy_decode(self, out, topk=1):
		"""

		:param out: output distribution over characters
		:param topk: top k predictions
		:return: predicted character, character_id, and probability
		"""
		if out is None:
			self.ended = True
			self.probability = -1

		if not self.ended:
			prob, char_id = torch.topk(out[:, -1], topk)
			next_char = Dataset.CHAR_ID_MAP[char_id.data[0]]

			self.characters += next_char
			self.probability *= torch.exp(prob).tolist()[0][0]
			self.targets = torch.cat((self.targets, torch.tensor([[char_id]]).to(self.device)), 1)

			if char_id == Dataset.EOS_ID:
				self.ended = True

			return next_char, char_id.data[0], prob
		else:
			return self.characters, None, self.probability

	def reversed_word(self):
		return self.characters[::-1]

	@staticmethod
	def beam_search_decoding(outputs, k):
		"""
		Beam search decoding for character sequence, not used. Needs more work to use this during validation
		:param outputs:
		:param k:
		:return:
		"""
		sequences = [[list(), 1.0]]
		# walk over each step in sequence
		for row in outputs:
			all_candidates = list()
			# expand each current candidate
			for i in range(len(sequences)):
				seq, score = sequences[i]
				for j in range(len(row)):
					candidate = [seq + [j], score * -log(row[j])]
					all_candidates.append(candidate)
			# order all candidates by score
			ordered = sorted(all_candidates, key=lambda tup: tup[1])
			# select k best
			sequences = ordered[:k]
		return sequences

	def strip_special_symbols(self):
		"""
		During training we learn the model to predict special characters. However, during training we should ignore them.
		Therefore, we remove all the special characters from the predicted output string.
		:return:
		"""
		self.characters = re.sub('[^A-Za-z0-9]+', '', self.characters)
