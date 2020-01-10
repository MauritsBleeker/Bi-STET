import seaborn
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec


def draw(data, x, y, ax):
	"""

	:param data: attention vectors
	:param x: x axis tick values
	:param y: y axis tick values
	:param ax: axis class
	:return:
	"""

	seaborn.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=torch.max(data), cbar=False, ax=ax, )


def draw_encoder_self_attention(attn_dict, image, w_ratio, w_headmap, h_headmap, h_ratio, n_heads,start_layer=0, end_layer=6, step_size=2):
	"""

	:param attn_dict:
	:param image:
	:param w_ratio:
	:param w_headmap:
	:param h_headmap:
	:param h_ratio:
	:param n_heads:
	:param start_layer:
	:param end_layer:
	:param step_size:
	:return:
	"""

	for layer in range(start_layer, end_layer, step_size):

		fig = plt.figure(figsize=((1 / w_ratio) * w_headmap, (1 / h_ratio) * h_headmap))
		gs = gridspec.GridSpec(2, n_heads + 1, width_ratios=[1 - n_heads * w_ratio] + [w_ratio] * n_heads, height_ratios=[h_ratio, 1 - h_ratio], wspace=0.01, hspace=0.01)

		print("Encoder self-attention layer", layer + 1)

		ax0 = plt.subplot(gs[0])
		ax0.imshow(np.flip(image.transpose(1, 0, 2), 1), aspect='auto')
		ax0.axis('off')
		ax0.axis('off')

		for h in range(n_heads):
			axs = plt.subplot(gs[h + 1])
			axs.set_title('Attention Head {} '.format(h + 1), fontsize=20)
			draw(attn_dict['encoder'][layer]['self_attn'][0, h], [], [], ax=axs)
			axs = plt.subplot(gs[h + n_heads + 2])
			axs.imshow(image, aspect='auto')
			axs.axis('off')

		fig.suptitle('Encoder self-attention layer {}'.format(layer + 1), fontsize=24)
		plt.tight_layout()
		plt.show()


def draw_decoder_self_attention(image, attn_dict, target_word, n_heads, h_headmap, w_headmap, h_ratio,start_layer=0, end_layer=6, step_size=2):
	"""

	:param attn_dict:
	:param target_word:
	:param n_heads:
	:param h_headmap:
	:param w_headmap:
	:param start_layer:
	:param end_layer:
	:param step_size:
	:return:
	"""

	for layer in range(start_layer, end_layer, step_size):
		fig = plt.figure(figsize=(n_heads * w_headmap * 2,  h_headmap * 2))
		gs = gridspec.GridSpec(1, n_heads, width_ratios=[1 / n_heads] * n_heads, height_ratios=[1], wspace=0.01, hspace=0.01)

		for h in range(n_heads):
			axs = plt.subplot(gs[h])
			axs.set_title('Attention Head {} '.format(h + 1), fontsize=34)
			draw(attn_dict['decoder'][layer]['self_attn'][0, h][:len(target_word), :len(target_word)], target_word, target_word if h == 0 else [], ax=axs)
		fig.suptitle('Decoder self-attention layer {}'.format(layer + 1), fontsize=44)
		plt.tight_layout()
		plt.show()

		fig = plt.figure(figsize=(n_heads * w_headmap * 2, h_headmap * 2))
		gs = gridspec.GridSpec(2, n_heads, width_ratios=[1 / n_heads] * n_heads, height_ratios=[0.5, 1 - 0.5], wspace=0.01,
		                       hspace=0.01)

		axs = plt.subplot(gs[0])
		axs.axis('off')
		for h in range(n_heads):
			axs = plt.subplot(gs[h])
			draw(attn_dict['decoder'][layer]['src_attn'][0, h][:len(target_word), :54], [],
			     target_word if h == 0 else [],
			     ax=axs)
			axs.set_title('Attention Head {} '.format(h + 1), fontsize=24)
			axs = plt.subplot(gs[h + n_heads])
			axs.imshow(image)
			axs.axis('off')

		fig.suptitle('Decoder source-attention layer {}'.format(layer + 1), fontsize=44)
		plt.tight_layout()
		plt.show()


def visualize_attention(attn_dict, target_word, image):
	"""
	Visualize the model attention
	:param attn_dict: model state after last forward pass
	:param target_word: target word which is predicted
	:param image: input image
	:return:
	"""

	w_headmap = 5
	h_headmap = w_headmap

	w_image = 1
	n_heads = 4

	t_length = w_headmap * n_heads + w_image

	w_ratio = ((t_length - w_image) / n_heads) / t_length
	h_ratio = 0.7

	seaborn.set(font_scale=2.8)

	draw_encoder_self_attention(attn_dict, image, w_ratio, w_headmap, h_headmap, h_ratio, n_heads, start_layer=0, end_layer=6, step_size=2)
	draw_decoder_self_attention(image, attn_dict, target_word, n_heads, h_headmap, w_headmap, h_ratio)
