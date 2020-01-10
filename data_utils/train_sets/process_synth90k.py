import os

OUTDIR = '../data/mnt/ramdisk/max/90kDICT32px'
DATA_DIR = '../data/mnt/ramdisk/max/90kDICT32px'

annotation_test_file = 'annotation_test.txt'
annotation_train_file = 'annotation_train.txt'
annotation_val_file = 'annotation_val.txt'
imlist_file = 'imlist.txt'
lexicon_file = 'lexicon.txt'

lexicon = open(os.path.join(DATA_DIR,   lexicon_file), 'r')
annotations_training = open(os.path.join(DATA_DIR,   annotation_train_file), 'r')

lexicon = lexicon.read().split()
idx_to_word = {}

for idx, word in enumerate(lexicon):
	idx_to_word[idx] = word

with open(os.path.join(OUTDIR, 'data_file.txt'), 'a') as annotation_file:
	for idx, annotations in enumerate(annotations_training):
		img_id, lex_idx = annotations.split()
		word = idx_to_word[int(lex_idx)]
		word_from_file_name = img_id.split('/')[-1].split('_')[1].lower()

		img_path = img_id
		line = img_path + " " + word + "\n"

		if not (word == img_id.split('/')[-1].split('_')[1].lower()):
			continue
		else:
			annotation_file.write(line)

		if idx % 10000 == 0:
			print ('10000 iters')