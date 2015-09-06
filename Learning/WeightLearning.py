import theano
import theano.tensor as T

import numpy as np

class L2_Regularization(object):
	def __init__(self, learning_rate=0.01, L2=0.001):
		self.learning_rate = learning_rate
		self.L2 = L2

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size

	def set_layer(self, layer):
		self.layer = layer

	def update(self):
		W = self.layer.W
		b = self.layer.b

		batch_size = self.batch_size
		
		d_W = self.layer.delta_W
		d_b = self.layer.delta_b

		learning_rate = self.learning_rate
		L2 = self.L2

		self.layer.W = W - d_W * learning_rate - (learning_rate * L2 * W) / batch_size
		self.layer.b = b - d_W * learning_rate