"""

Thanks to https://github.com/bgshih/aster for the source code

Published as: ASTER: An Attentional Scene Text Recognizer with Flexible Rectification
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018

"""

import os
import io
import re
import glob

from PIL import Image

DATA_DIR = '../data/ICDAR13'
CROP_MARGIN = 0.15
OUT_FOLDER = '/zfs/ilps-plex1/slurm/datastore/mbleeke1/ICDAR13/text_recognition'


def _is_difficult(word):
	assert isinstance(word, str)
	return not re.match('^[\w]+$', word)


def create_ic13():

	groundtruth_dir = os.path.join(DATA_DIR, 'Challenge2_Test_Task1_GT')
	groundtruth_files = glob.glob(os.path.join(groundtruth_dir, '*.txt'))

	count = 0
	difficult = 0
	with open(os.path.join(OUT_FOLDER, 'icdar13_annotations.txt'), 'w') as annotation_file:
		for groundtruth_file in groundtruth_files:
			image_id = re.match(r'.*gt_img_(\d+).txt$', groundtruth_file).group(1)
			image_rel_path = 'img_{}.jpg'.format(image_id)
			image_path = os.path.join(DATA_DIR, 'Challenge2_Test_Task12_Images', image_rel_path)
			image = Image.open(image_path)
			image_w, image_h = image.size

			with open(groundtruth_file, 'r') as f:
				groundtruth = f.read()

			matches = re.finditer(r'^(\d+),\s(\d+),\s(\d+),\s(\d+),\s\"(.+)\"$', groundtruth, re.MULTILINE)
			for i, match in enumerate(matches):
				bbox_xmin = float(match.group(1))
				bbox_ymin = float(match.group(2))
				bbox_xmax = float(match.group(3))
				bbox_ymax = float(match.group(4))
				tag = match.group(5)

				if _is_difficult(tag):
					difficult += 1
					continue

				if CROP_MARGIN > 0:
					bbox_h = bbox_ymax - bbox_ymin
					margin = bbox_h * CROP_MARGIN
					bbox_xmin = bbox_xmin - margin
					bbox_ymin = bbox_ymin - margin
					bbox_xmax = bbox_xmax + margin
					bbox_ymax = bbox_ymax + margin
				bbox_xmin = int(round(max(0, bbox_xmin)))
				bbox_ymin = int(round(max(0, bbox_ymin)))
				bbox_xmax = int(round(min(image_w - 1, bbox_xmax)))
				bbox_ymax = int(round(min(image_h - 1, bbox_ymax)))

				word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))

				image_name = "icdar13_cropped_image-" + str(count) + ".png"
				line = image_name + " " + tag.lower() + "\n"

				word_crop_im.save(os.path.join(OUT_FOLDER, image_name))

				annotation_file.write(line)
				annotation_file.flush()

				count += 1

	print('{} examples created'.format(count))
	print('{} difficult examples'.format(difficult))


if __name__ == '__main__':
	create_ic13()