import sys
import os
import glob
from shutil import copyfile

DATA_FOLDER = '../data/SynthText/cropped/'
ANNOTATION_FILE = 'annotations.txt'
PROJECT_DATA_FOLDER = '../data/'
ANNOTATION_FILE = 'annotations.txt'

if __name__ == '__main__':
	with open(DATA_FOLDER + ANNOTATION_FILE) as f:
		image_data = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	images = [data_line.strip()  for data_line in image_data[:1000] if len(data_line.strip().split(' ')[1]) >= 5]
	annotation_file = open(PROJECT_DATA_FOLDER + ANNOTATION_FILE, 'w+')
	for image_line in images:
		# copy to data folder make annotation file
		file_name, annotation = image_line.split(' ')
		copyfile(DATA_FOLDER + file_name, PROJECT_DATA_FOLDER + file_name)
		annotation_file.write(image_line)
