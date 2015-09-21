from scipy import ndimage
from scipy import misc

import theano

import numpy as np

class CharacterClass(object):

	def __init__(self, character, files, target_param, training, validation, testing):
		self.character = character
		self.target_param = target_param
		self.files = files

		self.training = training
		self.validation = validation
		self.testing = testing
		self.datasets = [training, validation, testing]

	def scale(self, imgv, size):
		# We then resize the image
		imresize = misc.imresize(imgv, (size, size), interp='bilinear', mode='P')
		# The next step is to super impose this onto some background
		# Or image, however we don't care to do that yet
		return imresize

	def get_dataset(self, size=32, sigma=None, out_len=62):
		dataset = []
		
		for ds_type in self.datasets:
			target = np.zeros((
						len(ds_type),
						out_len
					))

			target[:,self.target_param] = 1
			dataset.append(
				(
					self.t3_images(ds_type, size, sigma).astype(theano.config.floatX),
					target.astype(theano.config.floatX)
				)
			)
		return dataset

	def t3_images(self, ds_type, size=32, sigma=None):
		t3 = np.zeros(
			(
				len(ds_type),
				size,
				size
			),
			dtype=theano.config.floatX
		)

		for i in range(len(ds_type)):
			imgv = misc.imread(ds_type[i], flatten=True)
			imgv = self.scale(imgv, size)
			if sigma != None:
				imgv = ndimage.gaussian_filter(imgv, sigma)

			t3[i] = imgv

		return t3

	def reset(self):
		self.trainingIter = iter(self.training)
		self.validationIter = iter(self.validation)
		self.test = iter(self.test)