import os
from pathlib import Path
import pickle
from collections import defaultdict


class LexiconInference(object):
	"""
	class for lexicon inference
	"""
	def __init__(self):
		self.lexicon = None
		self.dict_keys = None
		self.accuracies = defaultdict(int)
		self._number_of_samples = None

	def inference(self, predicted_word, image_id, ground_truth):
		"""

		:param predicted_word:
		:return:
		"""
		max_distance = 1e10
		lexicon_prediction = None
		for dict_key in self.dict_keys:
			for word in self.lexicon[image_id][dict_key]:
				distance = self.levenshtein_distance(word, predicted_word)
				if distance < max_distance:
					lexicon_prediction = word
					max_distance = distance
			if lexicon_prediction == ground_truth:
				self.accuracies[dict_key] += 1

	def levenshtein_distance(self, s1, s2):
		"""
		:param s1:
		:param s2:
		:return:
		"""

		if len(s1) > len(s2):
			s1, s2 = s2, s1

		distances = range(len(s1) + 1)
		for i2, c2 in enumerate(s2):
			distances_ = [i2+1]
			for i1, c1 in enumerate(s1):
				if c1 == c2:
					distances_.append(distances[i1])
				else:
					distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
			distances = distances_
		return distances[-1]

	def has_lexicon(self, folder):
		"""

		:param folder: folder of the evaluation dataset
		:return:
		"""

		lexicon_file = Path(os.path.join(folder, 'lexicon.pickle'))
		if lexicon_file.is_file():
			self.lexicon = pickle.load(open(lexicon_file, 'rb'))
			self.dict_keys = list(self.lexicon[list(self.lexicon.keys())[-1]].keys())
			self.accuracies = defaultdict(int).fromkeys(self.dict_keys, 0)
			self._number_of_samples = len(self.lexicon.keys())
			return True
		return False

	def get_accuracies(self):
		"""

		:return: accuracy score
		"""
		return [(lexicon_name, float(n_correct) / self._number_of_samples)for lexicon_name, n_correct in self.accuracies.items()]
