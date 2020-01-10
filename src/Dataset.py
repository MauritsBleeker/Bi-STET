import os
import torch
import pickle
import numpy as np
import torch.utils.data as data
from six import BytesIO as IO
from PIL import Image


class Dataset(data.Dataset):
	"""
	Dataset class
	"""

	PAD_ID = 0
	GO_ID = 1
	EOS_ID = 2
	CHAR_ID_MAP = ['', '', ''] + list('0123456789abcdefghijklmnopqrstuvwxyz') + list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
	MAX_SEQUENCE_LENGTH = 24

	def __init__(self, root_folder, annotation_file, input_transform, load_pickle=False, validation_set=False):
		"""
		
		:param root_folder: root folder of the dataset
		:param annotation_file: file with the image ids and labels. Image id is the path from the root folder to the image, root/folder/xxx.img. Label is the word depecited in the image
		:param input_transform:  Transform function for every input image of the model
		:param load_pickle: If true, data is loaded from pickles
		:param validation_set: If true, this class contains validation data
		"""
		
		super(Dataset, self).__init__()
		
		self.root_folder = root_folder
		self.annotation_file = annotation_file
		self.labels = dict()
		self.list_IDs = list()
		self.input_transform = input_transform
		self.pickle_data = None
		self.load_pickle = load_pickle
		self.validation_set = validation_set
		self.vocab_per_image = None

		if self.load_pickle:
			self._process_pickle()
		else:
			self._process_annotation_file()

	def __len__(self):
		"""
		
		:return: number of samples in dataset
		"""
		
		return len(self.list_IDs)

	def __getitem__(self, index):

		"""

		:param index: index point to the next sample
		:return: training sample
		"""

		image_id = self.list_IDs[index]  # image path is the image ID

		if not self.load_pickle:
			label = self.labels[image_id]
			original_image = self._pil_loader(os.path.join(self.root_folder, image_id))
		else:
			original_image = Image.open(IO(self.pickle_data[image_id]['data']))
			label = self.pickle_data[image_id]['label']
			if isinstance(label, (bytes, bytearray)):
				label = label.decode('ascii')

		label = label.rstrip().lower()
		image = self.input_transform(original_image)

		ltr_targets, rtl_targets = self.convert_tokens(label)

		ltr_target_y = self.one_hot_targets(ltr_targets)
		rtl_target_y = self.one_hot_targets(rtl_targets)

		mask = self.make_mask(ltr_targets)

		sample = {
			'images': image,
			'ltr_targets': ltr_targets,
			'rtl_targets': rtl_targets,
			'ltr_targets_y': ltr_target_y,
			'rtl_targets_y': rtl_target_y,
			'labels': label,
			'ids': image_id,
			'masks': mask,
		}

		if self.validation_set:
			sample['original_image'] = np.array(original_image)
		return sample

	def convert_tokens(self, tokens):
		"""
		covert a string of tokens to a array of corresponding character id's, a => 13, b => 14 etc.
		:param tokens: string with tokens
		:return: np.array with the token id per index
		"""
		token_ids = [self.CHAR_ID_MAP.index(token) for token in tokens.rstrip()]
		reversed_tokes = list(reversed(token_ids))

		ltr = np.array(
			[self.GO_ID] + token_ids + [self.EOS_ID] + [self.PAD_ID] * (self.MAX_SEQUENCE_LENGTH - len(tokens.rstrip())), dtype=np.int32
		)

		rtl = np.array([self.GO_ID] + reversed_tokes + [self.EOS_ID] + [self.PAD_ID] * (self.MAX_SEQUENCE_LENGTH - len(tokens.rstrip())), dtype=np.int32
		)
		return torch.from_numpy(ltr),  torch.from_numpy(rtl)
	
	def one_hot_targets(self, y):
		"""
		convert characters indexes in the y target into one hot vector
		:param y: target vector, shape [1, max_sequence length] with the indexes of the characters
		:return: one hot representation [max_sequence, len(self.CHARMAP)]
		"""
		
		one_hot = np.zeros((self.MAX_SEQUENCE_LENGTH + 2, len(self.CHAR_ID_MAP)))
		one_hot[np.arange(self.MAX_SEQUENCE_LENGTH + 2), y] = 1
		
		return one_hot
	
	def make_mask(self, target):
		"""
		make a mask to mask all the padding symbols for the loss calculation
		:param target:
		:return: numpy array with mask
		"""
		
		mask = np.ones((self.MAX_SEQUENCE_LENGTH + 2, len(self.CHAR_ID_MAP)))
		mask[np.where(target == self.PAD_ID), :] = 0
		return torch.from_numpy(mask[:-1, :])
	
	def _process_annotation_file(self, sorted_data=False):
		"""
		read the annotation file and store the image id's in a list and the labels in a dictionary, image_id => label
		:param sorted_data: sort data based on word length
		:return:
		"""
		
		with open(os.path.join(self.root_folder, self.annotation_file)) as annotations:
			for line in annotations:
				image_id, annotation = line.split(' ')
				self.list_IDs.append(image_id)
				self.labels[image_id] = annotation.rstrip()
		if sorted_data:
			self.list_IDs = [
				label for image_id, label in sorted(list(self.labels.items()), key=lambda x: len(x[1]))
			]

	@staticmethod
	def _pil_loader(path):
		"""
		Pil image loader open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
		:param path:
		:return: pill image object in RGB
		"""
		
		with open(path, 'rb') as f:
			img = Image.open(f)
			return img.convert('RGB')

	def _process_pickle(self, sorted_data=False):
		"""
		processing pickle file
		:param sorted_data: If true, load all the images including annotations from pickle files!
		:return:
		"""

		self.pickle_data = pickle.load(open(os.path.join(self.root_folder, self.annotation_file), 'rb'))
		self.list_IDs = list(self.pickle_data.keys())

		if sorted_data:
			self.list_IDs = [
				label for image_id, label in sorted(list(self.labels.items()), key=lambda x: len(x[1]))
			]
