import os
ROOT_PATH = '../data/SynthText-Cropped'

filenames = [
	'0-200000/0-200000-annotations.txt',
	'200000-400000/200000-400000-annotations.txt',
	'400000-600000/400000-600000-annotations.txt',
	'600000-800000/600000-800000-annotations.txt',
	'800000-858749/800000-858749-annotations.txt'
]

print('Start process')
with open(os.path.join(ROOT_PATH, 'SynthText-Cropped-Annotations.txt'), 'w') as outfile:
	for fname in filenames:
		print('New file process')
		path = os.path.join(ROOT_PATH, fname)
		with open(path) as infile:
			for line in infile:
				outfile.write('/'.join(line.split('/')[9:]))

print('Done process')