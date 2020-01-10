from utils.utils_functions import make_folder
import os
import datetime
import time


class ExampleWriter(object):
	"""
	Write evaluation examples to a correct/incorrect file
	"""
	def __init__(self, config):
		"""

		:param config: Config class
		"""

		self.config = config
		self.example_path = 'validation-results-' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')

		self.correct_path = None
		self.wrong_path = None
		self.incorrect_example_file = None
		self.correct_example_file = None

	def add_new_dataset(self, dataset_name):
		"""

		:param dataset_name: name of the evaluation dataset
		:return:
		"""
		if self.incorrect_example_file is not None:
			self.incorrect_example_file.close()
		if self.correct_example_file is not None:
			self.correct_example_file.close()

		folder_path = os.path.join(self.config.OUPUT_FOLDER, self.example_path, dataset_name)
		make_folder(folder_path, '')

		self.incorrect_example_file = open(os.path.join(folder_path, 'incorrect_examples.txt'), 'w')
		self.correct_example_file = open(os.path.join(folder_path, 'correct_examples.txt'), 'w')

	def write_example(self, image_id, prediction, label):
		"""

		:param image_id: id of evaluation example
		:param prediction: method prediction
		:param label: ground truth label
		:return:
		"""
		if self.incorrect_example_file is None or self.correct_example_file is None:
			raise Exception('No files defined')

		line = ' '.join([image_id, prediction, label]) + '\n'
		if prediction != label:
			file = self.incorrect_example_file
		else:
			file = self.correct_example_file

		file.write(line)
		file.flush()
