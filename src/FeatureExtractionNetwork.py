import torch
import torch.nn as nn
import math


class FeatureExtractionNetwork(nn.Module):
	"""
	https://arxiv.org/pdf/1507.05717.pdf, VGG based feature extractor
	"""
	def __init__(self, d_model):
		"""
		
		:param d_model: visual embeddings dim
		"""
		self.d_model = d_model
		super(FeatureExtractionNetwork, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 1), stride=2)
		)

		self.layer4 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=(2, 1), stride=2)
		)

		self.layer5 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(kernel_size=(2, 1), stride=1)
		)
		
	def forward(self, x, source_mask=None):
		"""
		
		:param x:
		:param source_mask:
		:return:
		"""
		
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)

		return torch.squeeze(out * math.sqrt(self.d_model), dim=2).permute((0, 2, 1))
