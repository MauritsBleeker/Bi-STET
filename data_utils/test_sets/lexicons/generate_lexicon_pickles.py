import os
import xml.etree.ElementTree as ET
import pickle
import scipy.io
import copy
import re
import random
from collections import defaultdict

SVT_DATA_DIR = '../data/validation-sets/SVT/svt1'
SVT_OUT_DIR = '../data/validation-sets/SVT/text_recognition'

IIIT5K_DATA_DIR = '../data/validation-sets/IIIT5K/'
IIIT5K_OUT = '../data/validation-sets/IIIT5K/test'
MAT_FILE = 'testdata.mat'

ICDAR03_DATA_DIR = '../data/validation-sets/ICDAR03/SceneTrialTest'
ICDAR03_OUT = '../data/validation-sets/ICDAR03/SceneTrialTest'

lex_sizes = set(['50','1k', 'full'])

LEXICON_SIZE = 50

"""
We want a dictonary like this

{
	dataset : {
		image_id : { 
			50 :
			1K: 
		} 
	}
}

"""


def _is_difficult(word):
	assert isinstance(word, str)
	return len(word) < 3 or not re.match('^[\w]+$', word)


def _random_lexicon(lexicon_list, groundtruth_text, lexicon_size):
	lexicon = copy.deepcopy(lexicon_list)
	del lexicon[lexicon.index(groundtruth_text.lower())]
	random.shuffle(lexicon)
	lexicon = lexicon[:(lexicon_size-1)]
	lexicon.insert(0, groundtruth_text)
	return lexicon


def generate_icdar03_lexicon():
	"""

	:return:
	"""
	xml_root = ET.parse(os.path.join(ICDAR03_DATA_DIR,'words.xml')).getroot()
	count =0
	lex = []
	out = defaultdict(dict)
	for image_node in xml_root.findall('image'):
		for i, rect in enumerate(image_node.find('taggedRectangles')):
			tag = rect.find('tag').text.lower()
			if not _is_difficult(tag):
				lex.append(tag)

	for image_node in xml_root.findall('image'):
		for i, rect in enumerate(image_node.find('taggedRectangles')):
			tag = rect.find('tag').text.lower()
			if not _is_difficult(tag):
				rand_lex = _random_lexicon(lex, tag, LEXICON_SIZE)
				image_name = "icdar03_cropped_image-" + str(count) + ".png"
				out[image_name] = {'50': rand_lex, 'full': lex}
				count += 1

	pickle.dump(out, open(os.path.join(ICDAR03_OUT, 'lexicon.pickle'), 'wb'))
	print('created icdar03 lexicon!')


def _process_iiit5k_lex(lex):
	out = []
	for word in lex:
		out.append(word[0].lower())
	return out


def generate_iiit5k_lexicon():
	mat = scipy.io.loadmat(os.path.join(IIIT5K_DATA_DIR, MAT_FILE))
	small_lex = mat['testdata']['smallLexi'][0]
	medium_lex = mat['testdata']['mediumLexi'][0]
	img_names = mat['testdata']['ImgName'][0]

	out = defaultdict(dict)
	for i in range(len(img_names)):
		image_name = img_names[i][0]
		image_small_lex = _process_iiit5k_lex(small_lex[i][0])
		image_medium_lex = _process_iiit5k_lex(medium_lex[i][0])

		out[image_name] = {
			'50': image_small_lex,
			'1k': image_medium_lex
		}

	pickle.dump(out, open(os.path.join(IIIT5K_OUT, 'lexicon.pickle'), 'wb'))
	print('created iiit5k lexicon!')


def generate_svt_lexicon():
	count = 0
	lexicon_per_image = defaultdict(dict)
	test_xml_path = os.path.join(SVT_DATA_DIR, 'test.xml')
	xml_root = ET.parse(test_xml_path).getroot()
	for image_node in xml_root.findall('image'):
		lex = image_node.find('lex').text.split(',')
		lex = [word.lower() for word in lex]
		for i, rect in enumerate(image_node.find('taggedRectangles')):
			image_name = "svt_cropped_image-" + str(count) + ".png"
			lexicon_per_image[image_name] = {'50': lex }
			count += 1
	pickle.dump(lexicon_per_image, open(os.path.join(SVT_OUT_DIR, 'lexicon.pickle'),'wb'))
	print('created svt lexicon!')


if __name__ == "__main__":
	# generate_svt_lexicon()
	# generate_iiit5k_lexicon()
	generate_icdar03_lexicon()
