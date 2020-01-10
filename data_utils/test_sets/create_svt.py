"""

Thanks to https://github.com/bgshih/aster for the source code

Published as: ASTER: An Attentional Scene Text Recognizer with Flexible Rectification
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018

"""

import os
import xml.etree.ElementTree as ET
from PIL import Image


DATA_DIR = '../data/svt1'
OUT_FOLDER = '../data/SVT/text_recognition'
CROP_MARGIN = 0.05


def create_svt_subset():
	test_xml_path = os.path.join(DATA_DIR, 'test.xml')
	count = 0
	xml_root = ET.parse(test_xml_path).getroot()
	with open(os.path.join(OUT_FOLDER, 'svt_annotations.txt'), 'w') as annotation_file:
		for image_node in xml_root.findall('image'):
			image_rel_path = image_node.find('imageName').text
			image_path = os.path.join(DATA_DIR, image_rel_path)
			image = Image.open(image_path)
			image_w, image_h = image.size
			
			for i, rect in enumerate(image_node.find('taggedRectangles')):
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
					
				bbox_xmin = int(round(max(0, bbox_x)))
				bbox_ymin = int(round(max(0, bbox_y)))
				bbox_xmax = int(round(min(image_w - 1, bbox_x + bbox_w)))
				bbox_ymax = int(round(min(image_h - 1, bbox_y + bbox_h)))
				
				word_crop_im = image.crop((bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
				
				crop_name = '{}:{}'.format(image_rel_path, i)
				
				tag = rect.find('tag').text.lower()
				
				image_name = "svt_cropped_image-" + str(count) + ".png"
				line = image_name + " " + tag.lower() + "\n"
				
				word_crop_im.save(os.path.join(OUT_FOLDER, image_name))
				
				annotation_file.write(line)
				annotation_file.flush()
				
				
				count += 1
	
	print('{} examples created'.format(count))


if __name__ == '__main__':
	create_svt_subset()