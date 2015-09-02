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
			(filter_shape[0]),
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
		print "convolving man"
		print input.eval()

		results = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=self.filter_shape,
			image_shape=self.image_shape
		)

		print "Printed convolve man"
		print results.eval()
		print results.shape.eval()
		print self.b.shape.eval()
		self.output = T.tanh(results + self.b.dimshuffle('x',0,'x','x'))
		print "output via activation yo"
		print self.output.eval()
		return self.output

class Pool:
	def __init__(self, shape):
		self.shape = shape

	def compute(self, input):
		print "Pooling yo"
		print input.eval()
		print input.shape.eval()
		self.output = downsample.max_pool_2d(
			input=input,
			ds=self.shape,
			ignore_border=True
		)
		return self.output

class Cross_Entropy:
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

class L2_Regularization:
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

"""
With our fully connected layer our 2D image is converted
back to a 1D array which we apply a dot for our weights
and add bias.

:type rng: random number generator
:type n_in: scalar 1D array size of all elements
:type n_out: this is matrix of weight values per output
{output : weight array (convert 2d array to 1d array)}
"""
class FCLayer:
	def __init__(
		self,
		input,
		n_out,
		learn,
		W=None,
		b=None,
		cost=None,
		activation=T.tanh
    	):
		self.activation = activation
		rng = np.random.RandomState()
		self.cost = cost

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

		print "lin_output"
		print lin_output.eval()

		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)

		print "Shape of our output is"
		print self.output.shape[0].eval()

		self.activation_jacobian_W = theano.gradient.jacobian(self.output, self.W)
		self.activation_jacobian_b = theano.gradient.jacobian(self.output, self.b)

		print "Our jacobian shape is"
		#print self.activation_jacobian_W.shape.eval()
		print self.output.eval()

		self.params = [self.W, self.b]

		self.learn = learn.set_layer(self)

	def get_output(self, input):
		lin_output = T.dot(input, self.W) + self.b

		return (
			lin_output if activation is None
			else self.activation(lin_output)
		)

	"""
	Takes in our inputs and spits out a whole bunch of predictions
	and then returns a tuple consisting:

	(avg_cost, jacobian_W, jacobian_b)
	"""
	def get_avg_cost(self, y, inputs):
		# C = -(1/n)*Sum[yln(a) + (1-y)ln(1-a)]
		predictions = scan.theano(
			fn=lambda input: self.get_output(T.argmax(input)),
			sequences=[inputs]
		)
		costs = theano.scan(
			fn=lambda pred, y: y*T.log(pred) + (1-y)T.log(1-pred),
			sequences=[predictions, y]
		)[0]
		avg_cost = costs.mean(costs)

		return (
			avg_cost,
			theano.gradient.jacobian(avg_cost, self.W),
			theano.gradient.jacobian(avg_cost, self.b)
		)

	def backpropagate(self, delta_W, delta_b):
		self.delta_W = T.batched_dot(delta_W,self.activation_jacobian_W)
		self.delta_b = T.batched_dot(delta_b,self.activation_jacobian_b)
		
		# Because we passed in self, we can just
		# have our learning algorithm update the
		# bias and weights for us
		self.learn.update()
		return (self.delta_W, self.delta_b)