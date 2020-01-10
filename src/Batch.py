import torch
import itertools
from torch.autograd import Variable
from utils.utils_functions import subsequent_mask


class Batch:
	"""
	Batch class, Contains all the attributes for one training iteration
	"""
	def __init__(self, data, device, pad_id=0, bidirectional=False):
		"""
		
		:param data: list with dictionaries with all the data attributes for one training pass, each dict is from one dataset
		:param device: reference to GPU or CPU depending on the configurations
		:param: bidirectional: if True, decode sequence bidirectional
		:param pad_id: padding symbol id

		"""
		
		assert type(data) is list
		if len(data) > 1:

			self.images = torch.cat(tuple([data_set_data['images'] for data_set_data in data]), 0).to(device)
			self.ltr_targets = torch.cat(tuple([data_set_data['ltr_targets'] for data_set_data in data]), 0)[:, :-1].to(device)
			self.rtl_targets = torch.cat(tuple([data_set_data['rtl_targets'] for data_set_data in data]), 0)[:, :-1].to(device)
			self.ltr_targets_y = torch.cat(tuple([data_set_data['ltr_targets_y'] for data_set_data in data]), 0)[:, 1:].to(device)
			self.rtl_targets_y = torch.cat(tuple([data_set_data['rtl_targets_y'] for data_set_data in data]), 0)[:, 1:].to(device) if bidirectional else None
			self.targets_embedding_mask = self._make_std_mask(self.ltr_targets, pad_id).to(device)
			self.target_mask = torch.cat(tuple([data_set_data['masks'] for data_set_data in data]), 0).to(device)
			self.labels = list(itertools.chain.from_iterable([data_set_data['labels'] for data_set_data in data]))

		elif len(data) == 1:
			data = data[0]
			self.images = data['images'].to(device)
			self.ltr_targets = data['ltr_targets'][:, :-1].to(device)
			self.rtl_targets = data['rtl_targets'][:, :-1].to(device)
			self.ltr_targets_y = data['ltr_targets_y'][:, 1:].to(device)
			self.rtl_targets_y = data['rtl_targets_y'][:, 1:].to(device) if bidirectional else None
			self.targets_embedding_mask = self._make_std_mask(self.ltr_targets, pad_id).to(device)
			self.target_mask = data['masks'].to(device)
			self.labels = data['labels']
		else:
			raise Exception("Data is empty")

	@staticmethod
	def _make_std_mask(target, pad_id):
		"""
		Create a mask to hide padding and future words
		:param target: target tensor
		:param pad_id: id of the padding symbol
		:return: target mask, to mask all the predictions after the EOW_ID
		"""
	
		tgt_mask = (target != pad_id).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(target.size(-1)).type_as(tgt_mask.data))
		return tgt_mask
