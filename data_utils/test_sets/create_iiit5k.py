import os
import scipy.io

ROOT = '../data/IIIT5K'
FILE = 'testdata.mat'


mat = scipy.io.loadmat(os.path.join(ROOT, FILE))

print('done')

ground_truths = mat['testdata']['GroundTruth'][0]
img_names = mat['testdata']['ImgName'][0]

with open(os.path.join(ROOT, 'iiit5k_annotations.txt'), 'a') as annotation_file:
	for i in range(len(img_names)):
		image_name = img_names[i][0]
		ground_truth = ground_truths[i][0].lower()
		line = image_name + " " + ground_truth + "\n"
		annotation_file.write(line)
		print(line)
