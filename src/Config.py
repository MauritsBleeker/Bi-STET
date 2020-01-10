from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
import os
import logging
import _pickle as cPickle


class Config:
	"""
	All the parameters/settings of the entire project
	"""

	RANDOM_SEED = 42
	EXPERIMENT_NAME = 'final'
	ROOT_OUT_FOLDER = '../out'
	OUPUT_FOLDER = os.path.join(ROOT_OUT_FOLDER, EXPERIMENT_NAME)
	OUTFILE = 'bi-stet.cp'
	LOG_FNAME = 'logging.log'
	SAMPLES_PER_BATCH = 64

	def __init__(self):
		self.DEBUG = False
		# INPUT/OUTPUT PARAMETERS

		self.MAX_SEQ_LENGTH = 24
		self.IMAGE_SHAPE = (32, 224)

		self.INPUT_TRANSFORM = Compose([
			Resize(self.IMAGE_SHAPE),
			ToTensor(),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		# TRAIN DATASETS

		self.LOAD_PICKLE = True

		self.SYNHT90K_ROOT = '..data/mnt/ramdisk/max/90kDICT32px'
		self.SYNTH90K_ANNOTATIONS = 'Synth90k-Annotations.txt'

		self.SYNTH_TEXT_ROOT = '..data/mnt/ramdisk/max/90kDICT32px/SynthTextCropped'
		self.SYNTH_TEXT_ANNOTATIONS = 'SynthText-Cropped-Annotations.txt'

		self.SYNHT90K_LOCAL_PICKLE_ROOT = '../data/Synth90k/'
		self.SYNTH90K_LOCAL_PICKLE_ANNOTATIONS = 'synth90k.pickle'

		self.SYNHT90K_PICKLE_ROOT = '../data/Synth90k'
		self.SYNTH90K_PICKLE_ANNOTATIONS = 'synth90k.pickle'

		self.SYNTH_TEXT_PICKLE_ROOT = '../data/SynthTextCropped'
		self.SYNTH_TEXT_PICKLE_ANNOTATIONS = 'Synth-text-cropped.pickle'

		self.SYNTH_TEXT_EXTEDNED_PICKLE_ROOT = '/../data/SynthText-Cropped-Extended'
		self.SYNTH_TEXT_EXTENDED_PICKLE_ANNOTATIONS = 'synth-text-extended.pickle'

		self.LOCAL_DATA_ROOT = '../data/test_train'
		self.LOCAL_DATA_ANNOTATIONS = 'annotations.txt'

		self.LOG_FNAME = self.LOG_FNAME

		self.TRAIN_DATASETS = [
				(self.SYNTH_TEXT_EXTEDNED_PICKLE_ROOT, self.SYNTH_TEXT_EXTENDED_PICKLE_ANNOTATIONS),
				(self.SYNTH_TEXT_PICKLE_ROOT, self.SYNTH_TEXT_PICKLE_ANNOTATIONS),
				(self.SYNHT90K_PICKLE_ROOT, self.SYNTH90K_PICKLE_ANNOTATIONS),
				# (self.SYNHT90K_LOCAL_PICKLE_ROOT, self.SYNTH90K_PICKLE_ANNOTATIONS),
				# (self.SYNTH_TEXT_ROOT, self.SYNTH_TEXT_ANNOTATIONS, 'SynthText'),
				# (self.SYNHT90K_ROOT, self.SYNTH90K_ANNOTATIONS, 'Synth90K'),
				# (self.LOCAL_DATA_ROOT, self.LOCAL_DATA_ANNOTATIONS, 'local_test_dataset')
		]
	
		# TEST DATASETS

		self.IIIT5K_LOCAL_ROOT = '../data/validation-sets/IIIT5K'
		self.IIIT5K_ROOT = '../data/test/IIIT5K'
		self.IIIT5K_DATA_ANNOTATIONS = 'test/iiit5k_annotations.txt'

		self.ICDAR03_ROOT = '../data/validation-sets/ICDAR03/SceneTrialTest/text_recognition'
		self.ICDAR03_DATA_ANNOTATIONS = 'icdar03_annotations.txt'

		self.ICDAR03_ROOT_LOCAL =  '../data/validation-sets/ICDAR03/SceneTrialTest/text_recognition/'
		self.ICDAR03_DATA_ANNOTATIONS_LOCAL = 'icdar03_annotations.txt'

		self.ICDAR13_ROOT = '../data/validation-sets/ICDAR13/text_recognition'
		self.ICDAR13_DATA_ANNOTATIONS = 'icdar13_annotations.txt'

		self.ICDAR13_ROOT_LOCAL = '../data/validation-sets/ICDAR13/text_recognition/'
		self.ICDAR13_DATA_ANNOTATIONS_LOCAL = 'icdar13_annotations.txt'

		self.ICDAR15_ROOT = '../data/validation-sets/ICDAR15'
		self.ICDAR15_DATA_ANNOTATIONS = 'icdar15_annotations.txt'

		self.CUTE80_ROOT = '../data/validation-sets/CUTE80'
		self.CUTE80_DATA_ANNOTATIONS = 'cute80_annotations.txt'

		self.SVT_ROOT = '../data/validation-sets/SVT/text_recognition'
		self.SVT_DATA_ANNOTATIONS = 'icdar03_annotations.txt'

		self.SVTP_ROOT = '../data/validation-sets/SVTP'
		self.SVTP_DATA_ANNOTATIONS = 'svtp_annotations.txt'

		# OUTPUT FOLDERS

		self.EXPERIMENT_NAME = self.EXPERIMENT_NAME
		self.ROOT_OUT_FOLDER = self.ROOT_OUT_FOLDER
		self.OUPUT_FOLDER = self.OUPUT_FOLDER
		self.OUTFILE = self.OUTFILE

		# TRAINING PARAMS

		self.NUM_WORKERS = 0
		self.RANDOM_SEED = self.RANDOM_SEED
		self.SAMPLES_PER_BATCH = self.SAMPLES_PER_BATCH
		self.TRAIN_ITERATIONS = 500000
		self.SUMMARY_INTERVAL = 200
		self.STORE_ITERVAL = 5000
		self.START_ITERATION = 0

		# MODEL PARAMS
		self.BIDIRECTIONAL_DECODING = True
		self.N = 6
		self.D_MODEL = 512
		self.D_FF = 2048
		self.H = 8
		self.DROPOUT = 0.1
		self.USE_RESNET = True
		self.RESNET_LAYERS = [3, 4, 6, 6, 3]

		# OPTIMIZER

		self.LEARNING_RATE = 1

		self.SIZE_AVARAGE = True
		self.REDUCE_LOSS = False
		self.WARMUP = 8000
		self.FACTOR = 1

		# LEARNING RATE SCHEDUELER

		self.LR_MILESTONES = [150000, 300000, 400000]
		self.LR_GAMMA = 0.1

		self.LOAD_MODEL = True
		self.VALIDATE = True

		self.CONFIG_FNAME = 'config.cpkl'
		self.MODEL_FILE = None

		if self.MODEL_FILE is not None and not self.VALIDATE:
			assert self.START_ITERATION > 1
	
	def store_config(self, path):
		"""
		
		:param path:
		:return:
		"""
		
		f = open(os.path.join(path, self.CONFIG_FNAME), 'wb')
		cPickle.dump(self.__dict__, f)
		f.close()
	
	def load_config(self, path):
		"""
		
		:param path: path to load the config from
		:return:
		"""
		
		with open(os.path.join(path, self.CONFIG_FNAME), 'rb') as f:
			self.__dict__.update(cPickle.load(f))

	def print_config(self, print_on_std=True, store=False):
		"""
		:param print_on_std: If true print
		:param store: store the config in a .txt file
		:return:
		"""
		
		if store:
			f = open(os.path.join(Config.OUPUT_FOLDER, 'config.txt'), 'w')
			
		for key, value in self.__dict__.items():
			if not(key[:2] == '__'):
				line = str(key) + " : " + str(value)
				if print_on_std:
					logging.info(line)
				if store:
					f.write(line + "\n")
		if store:
			f.close()

	def set_validation_config(self):

		self.VALIDATE = True
		self.MODEL_FILE = 'bi-stet.cp'
		self.LOAD_MODEL = True
		self.EXPERIMENT_NAME = 'final'
		self.ROOT_OUT_FOLDER = '../out'

		self.OUPUT_FOLDER = os.path.join(self.ROOT_OUT_FOLDER, self.EXPERIMENT_NAME)
		self.LEXICON_INFERENCE = True
		self.SAMPLES_PER_BATCH = 1
		self.LOAD_PICKLE = False
		self.LOG_FNAME = 'validation.log'
		self.FILTER_DIFFICULT = False
		self.SHOW_EXAMPLES = False
		self.WRITE_EXAMPLES = True
		self.WORD_LENGTH_ACCURACY = True
		self.BIDIRECTIONAL_DECODING = True

		self.IIIT5K_LOCAL_ROOT = '/Users/mauritsbleeker/Documents/PhD/Github/Paper-implementations/STET/data/IIIT5K'

		self.VALIDATION_DATASETS = [
			# (self.IIIT5K_ROOT, self.IIIT5K_DATA_ANNOTATIONS, 'iiit5k'),
			# (self.ICDAR03_ROOT, self.ICDAR03_DATA_ANNOTATIONS, 'icdar03'),
			# (self.SVT_ROOT, self.SVT_DATA_ANNOTATIONS, 'svt'),
			# (self.ICDAR13_ROOT, self.ICDAR13_DATA_ANNOTATIONS, 'icdar13'),
			# (self.ICDAR15_ROOT, self.ICDAR15_DATA_ANNOTATIONS, 'icdar15'),
			# (self.CUTE80_ROOT, self.CUTE80_DATA_ANNOTATIONS, 'cute80'),
			# (self.SVTP_ROOT, self.SVTP_DATA_ANNOTATIONS, 'svtp'),
			# (self.ICDAR03_ROOT_LOCAL, self.ICDAR03_DATA_ANNOTATIONS_LOCAL, 'icdar03_local'),
			# (self.ICDAR13_ROOT_LOCAL, self.ICDAR13_DATA_ANNOTATIONS_LOCAL, 'icdar13_local'),
			(self.IIIT5K_LOCAL_ROOT, self.IIIT5K_DATA_ANNOTATIONS, 'iiit5k'),
		]
