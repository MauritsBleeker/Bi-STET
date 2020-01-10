import numpy as np
import matplotlib.pyplot as plt
from _collections import defaultdict


class WordLengthAccuracy(object):

	def __init__(self, config, dataset_name):
		"""

		:param config: Config class
		:param dataset_name: name of the evaluation dataset
		"""
		self.config = config
		self.correct = defaultdict(int)
		self.incorrect = defaultdict(int)
		self.dataset_name = dataset_name

	def add_example(self, word_length, is_correct):
		"""

		:param word_length: sequence length of the output word
		:param is_correct: True or False, True is predicted correct
		:return:
		"""
		if is_correct:
			self.correct[word_length] += 1
		else:
			self.incorrect[word_length] += 1

	def generate_plot(self):
		max_word_lengthts = max(list(self.correct.keys()) + list(self.incorrect.keys()))
		words_lengths = list(range(1, max_word_lengthts + 1))
		accuracy = [self.correct[word_length] / (self.correct[word_length] + self.incorrect[word_length]) if (self.correct[word_length] + self.incorrect[word_length]) > 0 else 0 for word_length
		            in words_lengths]
		plt.bar(np.array(words_lengths), np.array(accuracy), width=0.8, bottom=None, align='center', data=None)
		plt.ylabel('Accuracy')
		plt.xlabel('Word Length')
		plt.xticks(words_lengths)
		plt.title('Accuracy per word length')
		plt.show()
