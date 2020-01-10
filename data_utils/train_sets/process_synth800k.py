from __future__ import division
from __future__ import print_function

import logging
import os
import sys

import numpy as np
import scipy.io
import argparse
from PIL import Image
import re

np.random.seed(42)

DATA_FOLDER = '../data/SynthText'
OUT_FOLDER = '../data/SynthText-Cropped-Extended'

RESIZE_IMG = False  # resize input image
NUM_IMGS = 858750 - 1  # total images to process

CROPPED = True  # IF True, cropp text instaces from the image
GRAY_SCALE = False  # convert images to gray scale
RANDOM = False  # random shuffle the order of the images
START_INDEX = 0  # here to start the image processing
END_INDEX = 200000  # End image processing, so multiple processes can run in parralel
character_set = set( list('0123456789abcdefghijklmnopqrstuvwxyz') + list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))


def validate_tags(tag_list):
	for tag in tag_list:
		if not (len(tag) >= 2 and tag.isalnum()):
			return False
	return True


def process_annotations(wordBBs, img_id, tags, img_scale=1):
	"""
	function to generate the same
	:return:
	"""
	labels = []
	if wordBBs.ndim == 2:
		wordBBs = np.expand_dims(wordBBs, axis=2)
	
	for i in range(wordBBs.shape[-1]):
		attrib = {}
		
		bbox = wordBBs[:, :, i] * float(img_scale)
		
		attrib['x'] = np.min(bbox[0, :])
		attrib['y'] = np.min(bbox[1, :])
		attrib['width'] = np.max(bbox[0, :]) - attrib['x']
		attrib['height'] = np.max(bbox[1, :]) - attrib['y']
		attrib['vec'] = np.array([attrib['y'], attrib['x'], attrib['height'], attrib['width']], dtype=np.float32)
		attrib['lex'] = []
		attrib['tag'] = tags[i]
		attrib['img_id'] = img_id
		
		labels.append(attrib)
	
	return labels


if __name__ == '__main__':
	
	args = sys.argv[1:]
	
	parser = argparse.ArgumentParser()
	parser.prog = 'data parser Symth800k'
	parser.add_argument('--start-index', dest="start_index", default=0,
						type=int,
						help=('Start index of the images'))
	
	parser.add_argument('--end-index', dest="end_index", type=int, default=100000,
						help=('End index of the images'))
	
	parser.add_argument('--gray-scale', dest="gray_scale", action='store_true',
						help=('Convert images to gray scale'))
	
	parser.add_argument('--random', dest="random", action='store_true', default=RANDOM,
						help=('random shuffle data'))
	
	parser.add_argument('--cropped', dest="cropped", action='store_true', default=CROPPED,
						help=('Crop text instances from data'))
	
	parameters = parser.parse_args(args)
	
	process_out_folder = os.path.join(OUT_FOLDER, str(parameters.start_index) + '-' + str(parameters.end_index))
	logging_file_name = os.path.join(process_out_folder, 'Synth_text_cropped-' + str(parameters.start_index) + '-' + str(parameters.end_index) + '.log')


	if not os.path.exists(process_out_folder):
		os.makedirs(process_out_folder)


	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
		filename=logging_file_name)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	
	logging.info("Loading GT into memory")
	gt = scipy.io.loadmat(os.path.join(DATA_FOLDER, 'SynthText', 'gt.mat'))
	logging.info("Loaded GT into memory")

	assert parameters.cropped and not parameters.random

	if parameters.random:
		assert type(parameters.start_indexd) == int and type(parameters.end_index) == int
	
	if not parameters.random:
		logging.info('Start index: %i' % (parameters.start_index))
		logging.info('End index: %i' % (parameters.end_index))
	else:
		logging.info('Making random Dataset batches classes')
	
	img_name_to_idx = {}

	for idx, img_name in enumerate(gt['imnames'][0]):
		full_path = os.path.join(DATA_FOLDER, 'SynthText', img_name[0])
		img_name_to_idx[full_path] = idx
	
	if parameters.random:
		subset = np.random.choice(list(img_name_to_idx.keys()), NUM_IMGS, replace=False)
	else:
		subset = sorted(list(img_name_to_idx.keys()))
	
	if parameters.random:
		data_set = subset[0:int(1.0 * NUM_IMGS)]
	# train_set = subset[0:int(1.0*NUM_IMGS)]
	# val_set = subset[int(0.8*NUM_IMGS):int(0.0*NUM_IMGS)]
	# test_set = subset[int(0.9*NUM_IMGS):int(0.0*NUM_IMGS)]
	else:
		data_set = subset[parameters.start_index:parameters.end_index]

	dset_list = [(data_set, 'cropped', 'cropped')]  # (train_set,'train_set', 'train'), #(val_set,'validation_set', 'val'), (test_set,'test_set', 'test')]
	logging.info("Start making datasets \n")
	
	if parameters.cropped:
		file_name = str(parameters.start_index) + '-' + str(parameters.end_index) + '-annotations.txt'
		# add name of files, if cropped
		annotation_file = open(os.path.join(process_out_folder, file_name), 'a')
	
	instances = 0
	folder_number = 0  # number of the folder to save the image
	
	for data_set_tuple in dset_list:
		
		data_set = data_set_tuple[0]
		data_set_name = data_set_tuple[1]
		save_folder = data_set_tuple[2]
		
		images = []
		labels_per_image = []
		image_scales = []
		
		max_height = 0
		max_width = 0
		longest_word = 0
		vocab = set([])
		
		for iteration, file_name in enumerate(data_set):
			
			img_id = file_name.split('/')[-1]
			img_idx = img_name_to_idx[file_name]
			wordBBs = gt['wordBB'][0][img_idx]
			tags = [tag for str in gt['txt'][0][img_idx] for tag in str.split()]

			img_scale = 1

			labels = process_annotations(wordBBs, img_id, tags, img_scale)

			try:
				img = Image.open(os.path.join(DATA_FOLDER, file_name))
			except Exception as e:
				logging.error('False img: %s', file_name)
				continue

			for annotation in labels:
				tag = annotation['tag'].lower()

				if not all((c in character_set) for c in tag):
					continue

				y1 = annotation['vec'][0]
				x1 = annotation['vec'][1]
				y2 = annotation['vec'][0] + annotation['vec'][2]
				x2 = annotation['vec'][1] + annotation['vec'][3]

				try:
					if parameters.gray_scale:
						img2 = img.crop((x1, y1, x2, y2)).convert('LA')
					else:
						img2 = img.crop((x1, y1, x2, y2))

				except Exception as e:
					logging.error('False img: %s', file_name)
					continue

				width, height = img2.size
				if height > 32 and width > 30 and (height / width) < 1.2 and width < 800 and height < 500 and len(tag) <= 24:
					out_folder = os.path.join(process_out_folder, str(folder_number))
					if not os.path.exists(out_folder):
						os.makedirs(out_folder)
					out_file = os.path.join(out_folder, 'synth_text_' + str(instances) + '_' + str(re.sub(r'\W+', '',  tag.lower())) + '.png')

					line = out_file + " " + tag.lower() + "\n"

					try:
						img2.save(out_file)
					except SystemError:
						logging.error('False img: %s', file_name)
						continue
					
					annotation_file.write(line)
					annotation_file.flush()

					instances += 1

					if instances % 1000 == 0 and instances > 0:
						folder_number += 1
					if instances % 10000 == 0:
						logging.info('10.000 text instances processed')

					word_len = len(tag)
					width, height = img2.size

					if width > max_width:
						max_width = width
					if height > max_height:
						max_height = height
					if word_len > longest_word:
						longest_word = word_len
					if tag not in vocab:
						vocab.add(tag)

			if iteration % 10000 == 0:
				logging.info('Images processed: %i', iteration)
	
	logging.info('Max width: %i', max_width)
	logging.info('Max height: %i', max_height)
	logging.info('Longest word: %i', longest_word)
	logging.info('Vocab size: %i', len(vocab))
	logging.info('Number of instances: %i', instances)
	
	annotation_file.close()
