import os

ROOT_FILE = '../data/validation-sets/svt_per'
ANNOTATION_FILE = 'imagelist.txt'


with open(os.path.join(ROOT_FILE, ANNOTATION_FILE), 'r') as f:
	with open(os.path.join(ROOT_FILE, 'svtp_annotations.txt'), 'w') as annotation_file:
		for line in f:
			line = line.split(' ')
			file_path = line[0]
			word = line[1]

			line = file_path + ' ' + word.lower() + '\n'
			annotation_file.write(line)
			annotation_file.flush()
		annotation_file.close()
print('done')