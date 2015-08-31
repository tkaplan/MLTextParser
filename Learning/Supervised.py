import theano.tensor as T
import theano.tensor.signal.conv as conv
import theano.tensor.signal.downsample as downsample
import theano

import numpy as np

class Convolution:
	"""
	In: tuple of (images) 
	"""
	def __init__(self, filter_shape, image_shape, filters):
		self.W = filters
		self.b = theano.shared(
			value=np.zeros(
			(filter_shape[1]),
			dtype=theano.config.floatX
		), borrow=True)
		self.image_shape = image_shape
		self.filter_shape = filter_shape

	"""
	:type filters: 3rd or second rank tensor
	"""
	@classmethod
	def withFilters(cls, image_shape, filters):
		filter_shape = tuple(filters.shape.eval())
		return cls(filter_shape, image_shape, filters)

	"""
	We assume that filters is a shared or theano variable being passed in

	:type image_shape: tuple or list of length 4
    :param image_shape: (batch size, num input feature maps,
                             image height, image width)

	:param filter_shape: (num input feature maps, number of filters,
                              filter height, filter width)

	# Not used but maybe piped into a pool.
	:type poolsize: scalar
	"""
	@classmethod
	def withoutFilters(cls, filter_shape, image_shape, poolsize=4):
		rng = np.random.RandomState()

		fan_in = (numpy.prod(filter_shape[0] * filter_shape[2:]) / poolsize)

		# each unit in the lower layer recieves a gradient from:
		# "num output feature maps * filter height * filter width"
		# / pooling size
		fan_out = (numpy.prod(filter_shape[1:]) /
					numpy.prod(poolsize))

		W_bound = numpy.sqrt(6. / (fan_in + fan_out))

		filters = theano.shared(
			numpy.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		return cls(filter_shape, image_shape, filters)

	def compute(self, input):
		results = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=self.filter_shape,
			image_shape=self.image_shape
		)

		self.output = T.tanh(results + self.b.dimshuffle('x',0,'x','x'))
		return self.output

class Pool:
	def __init__(self, shape):
		self.shape = shape

	def compute(self, input):
		self.output = downsample.max_pool_2d(
			input=input,
			ds=self.shape,
			ignore_border=True
		)
		return self.output

"""
With our fully connected layer our 2D image is converted
back to a 1D array which we apply a dot for our weights
and add bias.

:type rng: random number generator
:type n_in: scalar 1D array size of all elements
:type n_out: this is matrix of weight values per output
{output : weight array (convert 2d array to 1d array)}
"""
class HL:
	def __init__(self, input, n_out, W=None, b=None, cost=LogisticRegression, activation=T.tanh):
		rng = np.random.RandomState()

		n_in = input.shape[0].eval()

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

		W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		self.params = [self.W, self.b]

	def backpropagate(self, delta):
		self.W

class LogisticRegression(object):
	"""Multi-class logistric regression class
	"""
	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression
		"""
		self.W = theano.shared(
			value=numpy.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)

		self.b = theano.shared(
			value=numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]
		self.input = input
		return self.y_pred

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

class MLP(object):
	"""Multi-Layer Perceptron class

	A multilayer perceptron is a feedforward artificial neural network model
	that has one layer or more of hidden units and nonlinear activations.
	Intermediate layers usually have as activation function tanh or the
	sigmoid function (defined here by a ''HiddenLayer'' class) whie the
	top layer is a softmax layer (defined here by a ''Logisticregression''
	class).
	"""

	def __init__(self, rng, input, n_in, n_hidden, n_out):
		"""Initialize the parameters for the multilayer perceptron

		"""
		self.hiddenLayer = HiddenLayer(
			rng=rng,
			input=input,
			n_out=n_hidden,
			activation=T.tanh
		)

		self.logRegressionLayer = LogisticRegression(
			input=self.hiddenLayer.output,
			n_out=n_out
		)

		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.logRegressionLayer.W).sum()
		)

		self.L2_sqr = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.logRegressionLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)

		self.errors = self.logRegressionLayer.errors

		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

		self.input = input