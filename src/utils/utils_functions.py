import torch
import logging
import os
import errno
import torch.nn.functional as F
import torch.nn as nn
import math
import glob
import copy
import numpy as np
from collections import defaultdict
from Dataset import Dataset
from torch.utils.data import DataLoader


def clones(module, N):
	"""
	Produce N identical copies of a module.
	:param module: PyTorch module, typically a layer
	:param N: Number of copies
	:return: List with module copies
	"""

	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
	"""
	Compute 'Scaled Dot Product Attention'
	:param query: Matrix with Query Embeddings [N, V, d_model]
	:param key: Matrix with Key Embeddings [N, V, d_model]
	:param value: Matrix with Value Embeddings [N, V, d_model]
	:param mask: mask used during decoding to hide future embeddings
	:param dropout: dropout value
	:return:
	"""
	
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


def char_idxs_to_word(char_idxs):
	"""
	map the predicted character idexes into characters
	:param char_idxs:
	:return:
	"""
	
	word = ''
	for char_idx in char_idxs:
		char = Dataset.CHAR_ID_MAP[char_idx]
		word += char
	return word

	
def print_outputs(out, ground_truth):
	"""
	Process the output probabilities and print them
	:param out: Predicted output scores, PyTorch Tensor [N, max_seq_length, number_of_characters]
	:param ground_truth: List with ground truth stings with length N
	:return: None
	"""
	for i, prediction in enumerate(out):
		word = ''
		for index_prediction in prediction:
			index_prediction = index_prediction.cpu()
			character_index = np.argmax(index_prediction.detach().numpy())
			char = Dataset.CHAR_ID_MAP[character_index]
			word += char
			if character_index == Dataset.EOS_ID:
				print("Predicted word: %s || Ground Truth: %s" % (word, ground_truth[i]))
				break
	

def get_dataset_loader(root_folder, annotation_file, input_transform, batch_size, shuffle=False, num_workers=1, load_pickle=False, validation_set=True):
	"""
	Get the Dataset Loader class
	:param root_folder: root folder where the data is located
	:param annotation_file: name of the annotation file
	:param input_transform: Transformation functions for the data
	:param batch_size: size of the batch
	:param shuffle: If true shuffle data
	:param num_workers:
	:param load_pickle: Boolean, if True load data from pickle files
	:param validation_set: True or False
	:return: DataLoader object
	"""
	dataset = Dataset(
		root_folder=root_folder,
		input_transform=input_transform,
		annotation_file=annotation_file,
		load_pickle=load_pickle,
		validation_set=validation_set,
	)
	
	dataset_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
	)
	
	return dataset_loader


def make_logger(config):
	"""
	make a global logger for the entire project
	:config: Config class
	:return:
	"""

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
		filename=os.path.join(config.OUPUT_FOLDER, config.LOG_FNAME)
	)

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	

def make_folder(path, folder_name):
	"""
	make new folder
	:param path: path where you want to create the folder
	:param folder_name: name of the folder
	:return:
	"""
	
	try:
		os.makedirs(os.path.join(path, folder_name))
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise Exception("Folder already exists")


def get_latest_check_point(folder):
	"""
	Return the newest model checkpoint file from a folder
	:param folder: folder where checkpoints are located
	:return: name of the last stored model
	"""
	
	list_of_files = glob.glob(os.path.join(folder, '*.cp'))
	latest_file = max(list_of_files, key=os.path.getctime)
	latest_file = latest_file.split('/')[-1]
	return latest_file


def subsequent_mask(size):
	"""
	Mask out subsequent positions.
	:param size:
	:return:
	"""

	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0


def concat_ltr_rtl(ltr, rtl):
	"""

	:param ltr: left-to-right targets
	:param rtl: right-to-left targets
	:return: concatenation
	"""
	if rtl is None:
		return ltr
	else:
		return torch.cat((ltr, rtl), 0)


def get_attention_distributions(encoder, decoder):
	"""
	Get a deep copy of the attention distribution, for analyses
	:param encoder: encoder states
	:param decoder: decoder states
	:return:
	"""
	out = defaultdict(list)
	for i, layer in enumerate(encoder.layers):
		self_attn_dist = copy.deepcopy(layer.self_attn.attn.data)
		out['encoder'].append({'self_attn': self_attn_dist})
	for i, layer in enumerate(decoder.layers):
		self_attn_dist = copy.deepcopy(layer.self_attn.attn.data)
		src_attn_dist = copy.deepcopy(layer.src_attn.attn.data)

		out['decoder'].append({
			'self_attn': self_attn_dist,
			'src_attn': src_attn_dist
		})

	return out
