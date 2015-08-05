from scipy import ndimage
from scipy import misc

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