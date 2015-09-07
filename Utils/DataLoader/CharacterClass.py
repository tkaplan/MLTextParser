from scipy import ndimage
from scipy import misc

import theano

import numpy as np

class CharacterClass(object):

	def __init__(self, character, files, training, validation, test):
		self.character = character
		self.files = files
		self.training = training
		self.validation = validation
		self.test = test
		self.training_it = iter(training)
		self.validation_it = iter(self.validation)
		self.test_it = iter(test)

	def scale(self, imgv, size):
		# We want to shorten by the largest dimmension
		resize = size / float(imgv.shape[0]) if imgv.shape[0] > imgv.shape[1] else size / float(imgv.shape[1]) 
		# We then resize the image
		imresize = misc.imresize(imgv, resize, 'bilinear', 'P')
		# The next step is to super impose this onto some background
		# Or image, however we don't care to do that yet
		return imresize

	def t3_training(self, pp, size=32, sigma = None):
		t3 = np.zeros(
			(
				len(self.training),
				32,
				32
			),
			dtype=theano.config.floatX
		)

		for i in range(len(self.training)):
			imgv = self.next_training()
			imgv = self.scale(size, imgv)
			if sigma != None:
				imgv = ndimage.gaussian_filter(imgv, sigma)

			t3[i] = imgv

		return t3

	def t3_validation(self, pp):
		t3 = np.zeros(
			(
				len(self.balidation),
				32,
				32
			),
			dtype=theano.config.floatX
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
			),
			dtype=theano.config.floatX
		)

		for _ in self.test:
			imgv = self.next_test()
			imgv = pp.scale(imgv)
			#imgv = pp.blur(imgv)
			t3 = np.append(t3,np.expand_dims(imgv,axis=0), axis=0)

		return t3

	def next_training(self):
		# Get file to load
		return misc.imread(next(self.training_it))

	def next_validation(self):
		# Get file to load
		return misc.imread(next(self.validation_it))

	def next_test(self):
		# Get file to load
		return misc.imread(next(self.test_it))

	def reset(self):
		self.trainingIter = iter(self.training)
		self.validationIter = iter(self.validation)
		self.test = iter(self.test)