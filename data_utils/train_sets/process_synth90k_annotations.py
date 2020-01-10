import os

DATA_FOLDER = '../data/mnt/ramdisk/max/90kDICT32px'

print('Start process')
with open(os.path.join(DATA_FOLDER, 'Synth90k-Annotations.txt'), 'w') as outfile:
	with open(os.path.join(DATA_FOLDER, 'data_file.txt')) as infile:
		for line in infile:
			line = line.split(' ')
			outfile.write(line[0][2:] + ' ' + line[1])

print('Done process')