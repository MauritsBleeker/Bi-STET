import torch
import torch.nn as nn
import math


class ResNet(nn.Module):
	
	"""
	https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8395027&tag=1
	https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

	ResNet Feature extractor
	"""

	def __init__(self, block, layers, d_model):
		"""

		:param block: ResNet Block
		:param layers: array with number of conv layers per ResNet layer
		:param d_model: model dimensionality
		"""
		
		super(ResNet, self).__init__()
		
		self.inplanes = 32
		self.d_model = d_model
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		
		self.block1 = self._make_layer(block, 32, layers[0], stride=(2, 2))
		self.block2 = self._make_layer(block, 64, layers[1], stride=(2, 2))
		self.block3 = self._make_layer(block, 128, layers[2], stride=(2, 1))
		self.block4 = self._make_layer(block, 256, layers[3], stride=(2, 1))
		self.block5 = self._make_layer(block, 512, layers[4], stride=(2, 1))

	def _make_layer(self, block, planes, blocks, stride):
		"""
		:param block:
		:param planes:
		:param blocks:
		:param stride:
		:return:
		"""
		
		downsample = None
		
		if stride[0] != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.EXPANSION, stride),
				nn.BatchNorm2d(planes * block.EXPANSION),
			)
		
		layers = list()
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.EXPANSION
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, image):
		"""
		
		:param image:
		:return:
		"""
		out = self.conv1(image)
		out = self.bn1(out)
		out = self.relu(out)
	
		out = self.block1(out)
		out = self.block2(out)
		out = self.block3(out)
		out = self.block4(out)
		out = self.block5(out)

		return torch.squeeze(out * math.sqrt(self.d_model), dim=2).permute((0, 2, 1))


class BasicBlock(nn.Module):
	
	"""
	BasicBlock ResNwr Block
	"""
	
	EXPANSION = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		"""
		
		:param inplanes:
		:param planes:
		:param stride:
		:param downsample:
		"""
		
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		"""
		
		:param x:
		:return:
		"""
		
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			residual = self.downsample(x)
		
		out += residual
		out = self.relu(out)

		return out
	

def conv3x3(in_planes, out_planes, stride=1):
	"""
	
	:param in_planes:
	:param out_planes:
	:param stride:
	:return:
	"""
	
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""
	
	:param in_planes:
	:param out_planes:
	:param stride:
	:return:
	"""
	
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
