import theano.tensor as T
import theano.tensor.signal.conv as conv
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

	"""
	:type filters: 3rd or second rank tensor
	"""
	@classmethod
	def withFilters(cls, image_shape, filters):
		print("####")
		print("####")
		print("####")
		print(filters.shape.eval())
		print("####")
		print("####")
		print("####")
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

		fan_in = (np.prod(filter_shape[0] * filter_shape[2:]) / poolsize)

		# each unit in the lower layer recieves a gradient from:
		# "num output feature maps * filter height * filter width"
		# / pooling size
		fan_out = (np.prod(filter_shape[1:]) /
					np.prod(poolsize))

		W_bound = np.sqrt(6. / (fan_in + fan_out))

		filters = theano.shared(
			np.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		filters = filters.reshape((
			filter_shape[0] * filter_shape[1],
			filter_shape[2],
			filter_shape[3]
		))

		return cls(tuple(filters.shape.eval()), image_shape, filters)

	def get_output(self, input):
		results = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=self.filter_shape,
			image_shape=self.image_shape
		)

		self.output = T.tanh(results + self.b.dimshuffle('x',0,'x','x'))
		return self.output

	def backpropagate(self, delta_W, delta_b):
		self.delta_W = T.batched_dot(delta_W,self.jacobian_W)
		self.delta_b = T.batched_dot(delta_b,self.jacobian_b)
		
		# Because we passed in self, we can just
		# have our learning algorithm update the
		# bias and weights for us
		self.learn.update()
		return (self.delta_W, self.delta_b)

class Pool(object):
	def __init__(self, shape):
		self.shape = shape

	def get_output(self, input):
		output = downsample.max_pool_2d(
			input=input,
			ds=self.shape,
			ignore_border=True
		)

		self.output = output.reshape((
			output.shape[0].eval() * output.shape[1].eval(),
			output.shape[2].eval(),
			output.shape[3].eval()
		))

		return self.output

	def backpropagate(self, delta_W, delta_b):
		self.delta_W = T.batched_dot(delta_W,self.jacobian_W)
		self.delta_b = T.batched_dot(delta_b,self.jacobian_b)
		
		# Because we passed in self, we can just
		# have our learning algorithm update the
		# bias and weights for us
		self.learn.update()
		return (self.delta_W, self.delta_b)

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
		learn=None,
		W=None,
		b=None,
		cost=None,
		activation=T.tanh
    	):
		self.activation = activation
		self.cost = cost
		#self.learn = learn.set_layer(self)

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

		W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
		self.params = [self.W, self.b]

	def get_output(self, input):
		lin_output = T.dot(input, self.W) + self.b

		self.output = (
			lin_output if self.activation is None
			else self.activation(lin_output)
		)

		return self.output

	"""
	Takes in our inputs and spits out a whole bunch of predictions
	and then returns a tuple consisting:

	(avg_cost, jacobian_W, jacobian_b)
	"""
	def get_cost(self, y, inputs):
		predictions = scan.theano(
			fn=lambda input: self.get_output(T.argmax(input)),
			sequences=[inputs]
		)

		costs = theano.scan(
			fn=self.cost,
			sequences=[predictions, y]
		)[0]

		self.avg_cost = costs.mean()

		self.grad_W = T.grad(avg_cost, self.W)
		self.grad_b = T.grad(avg_cost, self.b)

		return (
			self.avg_cost,
			self.grad_W,
			self.grad_b	
		)

	def backpropagate(self, delta_W, delta_b):
		self.delta_W = T.batched_dot(delta_W,self.grad_W)
		self.delta_b = T.batched_dot(delta_b,self.grad_b)
		
		# Because we passed in self, we can just
		# have our learning algorithm update the
		# bias and weights for us
		self.learn.update()
		return (self.delta_W, self.delta_b)