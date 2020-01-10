import os

ROOT_FILE = '../data/validation-sets/ICDAR15'
ANNOTATION_FILE = 'annotations.txt'
character_set = set( list('0123456789abcdefghijklmnopqrstuvwxyz') + list('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

def char_check(word):
  if not word.isalnum():
    return False
  else:
    for char in word:
      if char < ' ' or char > '~':
        return False
  return True

c = 0

with open(os.path.join(ROOT_FILE, ANNOTATION_FILE), 'r') as f:
	with open(os.path.join(ROOT_FILE, 'icdar15_annotations.txt'), 'w') as annotation_file:
		for line in f:
			try:
				image_name, word = line.split(', ')
			except:
				print(line)
			word = word.lower().strip().replace('"', '')
			if not all((c in character_set) for c in word) or not char_check(word):
				continue
			line = image_name + " " + word + "\n"
			c += 1
			#annotation_file.write(line)
			#annotation_file.flush()
print(c)