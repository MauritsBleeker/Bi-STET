import logging
import numpy as np
from utils.utils_functions import get_dataset_loader
from utils.visualization import visualize_attention
from utils.utils_functions import subsequent_mask
from utils.Word import Word
from utils.WordLengthAccuracy import WordLengthAccuracy
from utils.ExampleWriter import ExampleWriter
from utils.LexionInference import LexiconInference


class Validator(object):
	
	def __init__(self, model, dataset_paths, config, device):
		"""
		
		:param model: Vision Transformer
		:param dataset_paths: dataset paths is a list with triples (dataset_root_folder, annotation_file_name, datasetname)
		:param config: Config class
		:param device: GPU or CPU
		"""
		assert type(dataset_paths) is list
		
		self.model = model
		self.dataset_paths = dataset_paths
		self.config = config
		self.device = device
		self.datasets = {}
		self.get_datasets()

		if self.config.WRITE_EXAMPLES:
			self.example_writer = ExampleWriter(self.config)

	def validate(self, show_examples=False):
		"""
		:param show_examples: If true, log examples during validate
		:return:
		"""

		logging.info("Start validation")
		mean_validation_loss = []
		self.model.eval()

		for dataset_name, dataset in self.datasets.items():
			n_correct_ltr, n_correct_rtl, n_correct_bidirectional = 0, 0, 0
			n = 0

			if self.config.WRITE_EXAMPLES:
				self.example_writer.add_new_dataset(dataset_name)
			if self.config.WORD_LENGTH_ACCURACY:
				word_len_acc = WordLengthAccuracy(config=self.config, dataset_name=dataset_name)

			if self.config.LEXICON_INFERENCE:
				lexicon_inference = LexiconInference()
				has_lexicon = lexicon_inference.has_lexicon(dataset.dataset.root_folder)

			for _, example in enumerate(dataset):

				ltr_word = Word(self.device)
				rtl_word = Word(self.device)
				memory = self.model.encode(example['images'].to(self.device), src_mask=None)
				for char_index in range(self.config.MAX_SEQ_LENGTH):
					try:
						ltr, rtl = self.model.decode(memory,
							src_mask=None,
							ltr_targets=ltr_word.targets,
							rtl_targets=rtl_word.targets,
							ltr_tgt_mask=subsequent_mask(ltr_word.targets.shape[-1]).to(self.device),
							rtl_tgt_mask=subsequent_mask(rtl_word.targets.shape[-1]).to(self.device),
						)
					except RuntimeError:
						print('error')

					ltr_word.greedy_decode(ltr)
					rtl_word.greedy_decode(rtl)

					if ltr_word.ended and rtl_word.ended:
						ltr_word.strip_special_symbols()
						rtl_word.strip_special_symbols()

						if ltr_word.probability >= rtl_word.probability or not self.config.BIDIRECTIONAL_DECODING:
							predicted_word = ltr_word.characters
						else:
							predicted_word = rtl_word.reversed_word()

						if ltr_word.characters == example['labels'][0]:
							n_correct_ltr += 1
						accuracy_ltr = n_correct_ltr / float(n + 1)

						if self.config.BIDIRECTIONAL_DECODING:

							if rtl_word.reversed_word() == example['labels'][0]:
								n_correct_rtl += 1
							accuracy_rtl = n_correct_rtl / float(n + 1)

							if predicted_word == example['labels'][0]:
								n_correct_bidirectional += 1
							accuracy_bidirectional = n_correct_bidirectional / float(n + 1)

						correct = predicted_word == example['labels'][0]

						if show_examples and not correct:
							logging.info("Predicted: {} VS. Ground Truth: {}. Current accuracy: {}, N: {}".format(predicted_word, example['labels'][0], accuracy_ltr, n))
							visualize_attention(self.model.ltr_attn_dist, ['GO_ID'] + list(predicted_word), example['original_image'][0].numpy())
						if self.config.WORD_LENGTH_ACCURACY:
							word_len_acc.add_example(len(predicted_word), is_correct=correct)
						if self.config.WRITE_EXAMPLES:
							self.example_writer.write_example(example['ids'][0], predicted_word, example['labels'][0])

						if has_lexicon:
							lexicon_inference.inference(predicted_word, example['ids'][0], example['labels'][0])

						n += 1
						break
						
			logging.info("Test dataset: {}, left-to-right accuracy: {}".format(dataset_name, round(accuracy_ltr * 100, 1)))
			if self.config.BIDIRECTIONAL_DECODING:
				mean_validation_loss.append(accuracy_bidirectional)
				logging.info("Test dataset: {}, right-to-left accuracy: {}".format(dataset_name, round(accuracy_rtl * 100, 1)))
				logging.info("Test dataset: {}, bidirectional accuracy: {}".format(dataset_name, round(accuracy_bidirectional * 100, 1)))
			else:
				mean_validation_loss.append(accuracy_ltr)

			if self.config.LEXICON_INFERENCE:
				accuracies = lexicon_inference.get_accuracies()
				for lex_size, accuracy in accuracies:
					logging.info("Test dataset: {} with lexicon size {},  accuracy: {}".format(dataset_name, lex_size, round(accuracy * 100, 1)))

			logging.info("___________________________________________")
		return np.mean(mean_validation_loss)

	def get_datasets(self):
		"""
		
		:return:
		"""
		
		for dataset in self.dataset_paths:
			
			root_folder = dataset[0]
			annotation_file = dataset[1]
			dataset_name = dataset[2]
			dataset_loader = get_dataset_loader(root_folder=root_folder, annotation_file=annotation_file, input_transform=self.config.INPUT_TRANSFORM, batch_size=1, validation_set=True)
			self.datasets[dataset_name] = dataset_loader
