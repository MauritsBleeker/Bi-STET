
import io
from PIL import Image
from six import BytesIO as IO
import re
import os
import pickle


ROOT_DIR = '../data/SynthText-Cropped-Extended/600000-800000'
ANNOTATION_FILE = '600000-800000-annotations.txt'
data = dict()

if __name__ == "__main__":
	print('start process')

	with open(os.path.join(ROOT_DIR, ANNOTATION_FILE), 'r') as annotations:
		for idx, line in enumerate(annotations):
			line = line.rstrip('\n')

			img_path, label = line.split(' ')

			try:

				img = Image.open(os.path.join(ROOT_DIR, img_path), mode='r')

				imgByteArr = io.BytesIO()
				img.save(imgByteArr, format='PNG')
				imgByteArr = imgByteArr.getvalue()

				data[img_path] = {'data': imgByteArr, 'label': label}

			except Exception:
				print('wrong image, ignoring line %i: %s', idx + 1, line)
				continue

			if idx % 5000 == 0 and idx > 0:
				print('%i done'%idx)

	with open(os.path.join(ROOT_DIR, 'synth800k.pickle'), 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('done')
