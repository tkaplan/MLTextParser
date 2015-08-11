from scipy import ndimage
from scipy import misc

import numpy as np

class CharacterClass():

	def __init__(self, character, files, training, validation, test):
		self.character = character
		self.files = files
		self.training = training
		self.validation = validation
		self.test = test
		self.training_it = iter(training)
		self.validation_it = iter(self.validation)
		self.test_it = iter(test)

	def t3_training(self, pp):
		t3 = np.zeros(
			(
				len(self.training),
				32,
				32
			)
		)

		for _ in self.training:
			imgv = self.next_training()
			imgv = pp.scale(imgv)
			imgv = pp.blur(imgv)
			t3 = np.append(t3,np.expand_dims(imgv,axis=0), axis=0)

		return t3

	def t3_validation(self, pp):
		t3 = np.zeros(
			(
				len(self.balidation),
				32,
				32
			)
		)

		for _ in self.validation:
			imgv = self.next_validation()
			imgv = pp.scale(imgv)
			imgv = pp.blur(imgv)
			t3 = np.append(t3,np.expand_dims(imgv,axis=0), axis=0)

		return t3

	def t3_test(self, pp):
		t3 = np.zeros(
			(
				len(self.test),
				32,
				32
			)
		)

		for _ in self.test:
			imgv = self.next_test()
			imgv = pp.scale(imgv)
			imgv = pp.blur(imgv)
			t3 = np.append(t3,np.expand_dims(imgv,axis=0), axis=0)

		return t3

	def next_training(self):
		# Get file to load
		return misc.imread(self.training_it.next())

	def next_validation(self):
		# Get file to load
		return misc.imread(self.validation_it.next())

	def next_test(self):
		# Get file to load
		return misc.imread(self.test_it.next())

	def reset(self):
		self.trainingIter = iter(self.training_it)
		self.validationIter = iter(self.validation_it)
		self.test = iter(self.test_it)