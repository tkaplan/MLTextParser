from os import listdir
import os.path as path
import Utils.DataLoader.CharacterClass as cc
import random

class NOBase(object):
	# training: 0 < float < 1
	# validation: 0 < float < 1
	# training + validation =< 1
	# test set is 1 - training - validation
	def __init__(self, training, validation):
		# training, validation, and test should be normalized ratios
		if (0 < training < 1) is False:
			raise Exception("Training must be 0 < t < 1")
		self.training = training

		if (0 < validation < 1) is False:
			raise Exception("Validation must be 0 < v < 1")
		self.validation = validation

		if training + validation > 1:
			raise Exception("Training + Validation must be less than 1")
		self.test = 1 - self.training - validation
		
		self.ascii_ranges = {
			'z': ord('z'),
			'a': ord('a'),
			'A': ord('A'),
			'Z': ord('Z'),
			'0': ord('0'),
			'8': ord('8')
		}
		return

	def get_classno(self, character):
		ascii = ord(character)
		if self.ascii_ranges['z'] >= ascii >= self.ascii_ranges['a']:
			return ascii - 60
		if self.ascii_ranges['Z'] >= ascii >= self.ascii_ranges['A']:
			return ascii - 54
		if self.ascii_ranges['8'] >= ascii >= self.ascii_ranges['0']:
			return '0%s' % (ascii - 47)
		if character == '9':
			return '10'
		raise Exception("Character not recognized")

	def split_datasets(self, files):
		training = random.sample(files, int(self.training * len(files)))
		newSet =  set(files) - set(training)
		validation = random.sample(newSet, int(self.validation * len(newSet)))
		test = newSet - set(validation)
		return (list(training), list(validation), list(test))

	def get_characterset(self, character):
		# Get number of files in folder
		folder = self.get_classpath(character)
		
		# Files we need to load
		files = [ path.join(folder,f) for f in listdir(folder) if path.isfile(path.join(folder,f)) ]
		training, validation, test = self.split_datasets(files)
		return cc.CharacterClass(
			character = character,
			files = files,
			training = training,
			validation = validation,
			test = test
		)







