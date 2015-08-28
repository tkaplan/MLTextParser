import logging

def logger(name):
	file_handler = logging.FileHandler(name)
	file_handler.setLevel(logging.INFO)
	fh_fmt = logging.Formatter("%(asctime)s (%(levelname)s)\t: %(message)s")
	file_handler.setFormatter(fh_fmt)
	logger = logging.getLogger(name)
	logger.addHandler(file_handler)
	logger.setLevel(logging.INFO)
	return logger