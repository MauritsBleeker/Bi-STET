import numpy as np
import torch
from Config import Config
from Trainer import Trainer
from Validator import Validator
from BiSTET import BiSTET
from utils.utils_functions import make_logger, make_folder, get_latest_check_point
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
random.seed(Config.RANDOM_SEED)
torch.cuda.manual_seed_all(Config.RANDOM_SEED)

cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

if __name__ == '__main__':

	config = Config()
	make_folder(Config.ROOT_OUT_FOLDER, Config.EXPERIMENT_NAME)
	make_logger(Config)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	bi_stet = BiSTET(
		config=config,
		device=device,
	)
	
	bi_stet.model.to(device)
	
	if config.LOAD_MODEL:
		if config.MODEL_FILE is None:
			config.MODEL_FILE = get_latest_check_point(
				folder=config.OUPUT_FOLDER
			)
	
		bi_stet.load_model(
			path=config.OUPUT_FOLDER,
			file=config.MODEL_FILE
		)

	validator = Validator(
		model=bi_stet.model,
		dataset_paths=config.VALIDATION_DATASETS,
		config=config,
		device=device
	)

	if not config.VALIDATE:

		config.store_config(path=Config.OUPUT_FOLDER)
		config.print_config(store=True)
		trainer = Trainer(
			model=bi_stet,
			config=config,
			device=device,
		)
		
		trainer.train()

	else:

		config.print_config(store=False)
		validator.validate(show_examples=config.SHOW_EXAMPLES)
