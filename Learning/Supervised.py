import theano.tensor as T
from theano.tensor.nnet import conv
import theano.tensor.signal.downsample as downsample
import theano

import numpy as np

class Convolution(object):
	"""
	In: tuple of (images) 
	"""
	def __init__(self, filter_shape, image_shape, filters):
		self.W = filters
		self.b = theano.shared(
			value=np.zeros(
			(filter_shape[0]),
			dtype=theano.config.floatX
		), borrow=True)
		self.image_shape = image_shape
		self.filter_shape = filter_shape
		self.params = [self.W, self.b]

	def set_zactivator(self, alpha=.5):
		return None
	"""
	:type filters: 3rd or second rank tensor
	"""
	@classmethod
	def withFilters(cls, filter_shape, image_shape, filters):
		filter_shape = filter_shape
		return cls(filter_shape, image_shape, filters)

	"""
	We assume that filters is a shared or theano variable being passed in

	:type image_shape: tuple or list of length 4
    :param image_shape: (batch size, num input feature maps,
                             image height, image width)

	:param filter_shape: (num input filters, number of feature maps,
                              filter height, filter width)

	# Not used but maybe piped into a pool.
	:type poolsize: scalar
	"""
	@classmethod
	def withoutFilters(cls, filter_shape, image_shape, poolsize=4):
		rng = np.random.RandomState()

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
			np.prod(poolsize))

		W_bound = np.sqrt(6. / (fan_in + fan_out))

		filters = theano.shared(
			np.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		return cls(filter_shape, image_shape, filters)

	def output_shape(self):
		return (
			self.image_shape[0],
			self.filter_shape[1],
			self.image_shape[2] - self.filter_shape[2],
			self.image_shape[3] - self.filter_shape[3]
		)

	def get_output(self, input):
		results = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=self.filter_shape,
			image_shape=self.image_shape
		)

		self.output = T.tanh(results + self.b.dimshuffle('x',0,'x','x'))
		return self.output

class Pool(object):
	def __init__(self, shape):
		self.shape = shape

	def output_shape(self, image_shape, filter_shape):
		return (
			image_shape[0],
			filter_shape[0],
			int((image_shape[2] - filter_shape[2]) / self.shape[0]),
			int((image_shape[3] - filter_shape[3]) / self.shape[1])
		)

	def get_output(self, input):
		output = downsample.max_pool_2d(
			input=input,
			ds=self.shape,
			ignore_border=True
		)
		return output
"""
With our fully connected layer our 2D image is converted
back to a 1D array which we apply a dot for our weights
and add bias.

:type rng: random number generator
:type n_in: scalar 1D array size of all elements
:type n_out: this is matrix of weight values per output
{output : weight array (convert 2d array to 1d array)}
"""
class FCLayer(object):
	def __init__(
		self,
		n_in,
		n_out,
		W=None,
		b=None,
		activation=T.tanh
    	):
		self.activation = activation

		rng = np.random.RandomState()

		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)

			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

		self.W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, name='b', borrow=True)

		self.params = [self.W, self.b]

	def output_shape(self):
		return (n_in, n_out)

	def get_output(self, input):
		z = T.dot(input, self.W) + self.b

		self.output = (
			z if self.activation is None
			else self.activation(z)
		)

		return self.output