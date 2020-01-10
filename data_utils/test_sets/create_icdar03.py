"""

Thanks to https://github.com/bgshih/aster for the source code

Published as: ASTER: An Attentional Scene Text Recognizer with Flexible Rectification
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018

"""

import os
import copy
import random
import re
import xml.etree.ElementTree as ET
from PIL import Image


DATA_DIR = '../data/ICDAR03/SceneTrialTest'
IGNORE_DIFFICULT = True
CROP_MARGIN = 0.0
OUT_FOLDER = '/zfs/ilps-plex1/slurm/datastore/mbleeke1/ICDAR03/SceneTrialTest/text_recognition'

lexicon_size = 50
random.seed(1)


def _random_lexicon(lexicon_list, groundtruth_text, lexicon_size):
	lexicon = copy.deepcopy(lexicon_list)
	del lexicon[lexicon.index(groundtruth_text.lower())]
	random.shuffle(lexicon)
	lexicon = lexicon[:(lexicon_size-1)]
	lexicon.insert(0, groundtruth_text)
	return lexicon


def _is_difficult(word):
  assert isinstance(word, str)
  return len(word) < 3 or not re.match('^[\w]+$', word)


def create_ic03():

	xml_path = os.path.join(DATA_DIR, 'words.xml')
	xml_root = ET.parse(xml_path).getroot()
	count = 0
	difficult = 0
	
	with open(os.path.join(OUT_FOLDER, 'icdar03_annotations.txt'), 'w') as annotation_file:
		for image_node in xml_root.findall('image'):
			image_rel_path = image_node.find('imageName').text
			image_path = os.path.join(DATA_DIR, image_rel_path)
			image = Image.open(image_path)
			image_w, image_h = image.size
		
			for i, rect in enumerate(image_node.find('taggedRectangles')):
				tag = rect.find('tag').text.lower()
				if IGNORE_DIFFICULT and _is_difficult(tag):
					difficult += 1
					continue
				
				bbox_x = float(rect.get('x'))
				bbox_y = float(rect.get('y'))
				bbox_w = float(rect.get('width'))
				bbox_h = float(rect.get('height'))
				
				if CROP_MARGIN > 0:
					margin = bbox_h * CROP_MARGIN
					bbox_x = bbox_x - margin
					bbox_y = bbox_y - margin
					bbox_w = bbox_w + 2 * margin
					bbox_h = bbox_h + 2 * margin
					
				bbox_xmin = int(round(max(0., bbox_x)))
				bbox_ymin = int(round(max(0., bbox_y)))
				bbox_xmax = int(round(min(image_w-1., bbox_x + bbox_w)))
				bbox_ymax = int(round(min(image_h-1., bbox_y + bbox_h)))
				
				word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
				
				image_name = "icdar03_cropped_image-" + str(count) + ".png"
				line = image_name + " " + tag.lower() + "\n"
				
				word_crop_im.save(os.path.join(OUT_FOLDER, image_name))
				
				annotation_file.write(line)
				annotation_file.flush()
				
				count += 1
				
	print('{} examples created, total: {}'.format(count, difficult + count))


if __name__ == '__main__':
	create_ic03()
