import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessing
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

from Learning.Supervised import Convolution, Pool, FCLayer
import Learning.Supervised as supervised_learning
import Learning.Unsupervised as usupervised_learning
import Learning.Costs as costs

import numpy as np

import timeit

# self.nh = NOHandwritting.NOHandwritting(training, validation)
# char_classes = list(string.letters) + list(range(10))
# self.classes_size = len(char_classes)
# self.char_set = [self.nh.get_characterset(char) for char in char_classes]


# Assume we only classify fonts
class NeuralNetwork(object):
	def __init__(self,
		training,
		validation,
		test,
		batch_size=600,
		img_shape=(32,32),
		pre_processor=None
		):
		self.batch_size
		self.training = training
		self.validation = validation
		self.test = test
		self.layers = []
		self.params = []
		self.output = None
		self.finalized = False
		self.logger = dev_logger.logger(__name__ + ".NeuralNetwork")

	def build_fclayer(self, layer, previous_layer, last_output, **kwargs):
		# We need to reshape the last_output
		# depending on what type of layer we had
		if previous_layer['name'] in ['Pool', 'Convolution']:
			last_output = last_output.flatten(2)
			kwargs['n_in'] = (
				previous_layer['output_shape'][2] * \
				previous_layer['output_shape'][3]
			)
		else:
			# We can assume we have an fc layer
			kwargs['n_in'] = previous_layer['output_shape'][1]

		# With FCLayer we can just passing kwargs
		entity = FCLayer(**kwargs)
		layer['output_shape'] = entity.output_shape()

		return (layer, entity)

	def build_convolution(self, layer, previous_layer, last_output, **kwargs):
		input_shape = previous_layer['output_shape']
		layer['filter_shape'] = (
			kwargs['n_kerns'],
			input_shape[0],
			kwargs['height'],
			kwargs['width']
		)

		entity = Convolution.withoutFilters(
			filter_shape=layer['filter_shape'],
			image_shape=input_shape
		)
		
		layer['output_shape'] = entity.output_shape()
		return (layer, entity)

	def build_pool(self, layer, previous_layer, last_output, **kwargs):
		entity = Pool(kwargs['shape'])
		if previous_layer.name == 'Convolution':
			layer['output_shape'] = entity.output_shape(
				previous_layer['output_shape'],
				previous_layer['filter_shape']
			)
		else:
			layer['output_shape'] = (
				previous_layer['output_shape'] / kwargs['shape']
			)
		return (layer, entity)

	# We are assuming that Pool layer cannot be
	# first.
	#
	# First layers can either be convolution or fc.
	# If the first layer is a convolution, we require
	# 4-Rank Tensor otherwise we use a matrix of flattened
	# array images.
	def head_layer(self, layer, **kwargs):
		input = kwargs['input']
		# We must delete kwargs input because
		# we're going to use kwargs directly to
		# create our layer
		del kwargs['input']

		class_ = getattr(supervised_learning, layer['name'])
		entity = class_(**kwargs)
		layer['output_shape'] = entity.output_shape()

		if layer['name'] == 'Convolution':
			layer['filter_shape'] = entity.filter_shape
			self.input = T.ftensor4('inputs')
		else:
			self.input = T.fmatrix('inputs')

		return (layer, entity)


	# We assume all inputs/ outputs are t3
	def add(self, name, **kwargs):
		if self.finalized:
			raise Exception("You must reset the nerual net before adding layers")

		layer = {
			'name': name
		}

		previous_layer = self.layers[-1]
		last_output = previous_layer['outputs']

		if len(self.layers) == 0:
			layer, entity = self.head_layer(name, **kwargs)
			self.params += entity.params

		elif name == 'FCLayer':
			layer, entity = self.build_fclayer(layer, previous_layer, last_output, **kwargs)
			self.params += entity.params
		
		# Get the filter shape that we need for conv nets
		elif name == 'Convolution':
			layer, entity = self.build_convolution(layer, previous_layer, last_output, **kwargs)
			self.params += entity.params

		elif name == 'Pool':
			layer, entity = self.build_pool(layer, previous_layer, last_output, **kwargs)

		layer['outputs'] = entity.get_outputs(last_output)
		layer['entity'] = entity
		self.layers.push(layer)

	# This will transform our inputs before it's fed
	# into the net
	def set_preprocessor(self, preprocessor):
		self.preprocessor = preprocessor

	# We could add the option of choosing the end classifier,
	# but softmax is the easiest way to normalize and
	# binary_crossentropy seems like the best method that I
	# know of.
	def compile(self, n_out):
		# We should crash and burn if someone trys to compile
		# without any layers.
		if len(self.layers) == 0:
			raise Exception("Cannot compile a neural network without layers")

		# finalize our neural net so we can't add anymore layers
		self.finalized = True

		# get our output
		self.add('FCLayer', n_out=n_out, activation=T.nnet.softmax)

		output = self.layers[-1]
		
		# Define our input
		self.softmax_classify = theano.function(
			inputs=[self.input],
			outputs=[output]
		)

	# Returns a tuple of: (argmax(softmax), (sm0, sm1, ... smN))
	def softmax_classify(self, input):
		if self.preprocessor != None:
			input = self.preprocessor(input)

		return self.softmax_classify(input)
	
	# Set our training, testing, and validation data
	# Where data is organized as [(input, target)]
	def set_ttv_data(self, training, testing, validation):
		self.training = training
		self.testing = testing
		self.validation = validation

	def train(self,
		batch_size,
		learning_rate=.001,
		patience=10000,
		patience_increase=2,
		improvement_threshold=0.995
		):
		targets = T.fmatrix('targets')

		outputs = self.layers[-1]['outputs']
		# Build out our training model
		cost = T.mean(
			T.nnet.binary_crossentropy(
				outputs, targets
			)
		)
		
		params = self.params
		grads = T.grad(cost, params)

		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		index = T.lscalar()
		inputs_shared = self.training['inputs']
		targets_shared = self.training['targets']

		self.training_model = theano.function(
			inputs=[index],
			outputs=cost,
			updates=updates,
			givens={
				inputs: inputs_shared[index * batch_size: (index + 1) * batch_size],
				targets: targets_shared[index * batch_size: (index + 1) * batch_size]
			}
		)

	def train(self,
		patience=10000,
		patience_increase=2,
		improvement_threshold=0.995
	):
		print("... training the model")

		validation_frequency = min(n_train_batches, patience / 2)

		best_validation_loss = np.inf
		best_iter = 0
		test_score = 0.
		start_time = timeit.default_timer()

		done_looping = False
		epoch = 0
		while (epoch < n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in range(n_train_batches):
				minbatch_avg_cost = train_model(minibatch_index)
				iter = (epoch - 1) * n_train_batches + minibatch_index
				if (iter + 1) % validation_frequency == 0:
					validation_losses = [
						validate_model(i) for i in xrange(n_valid_batches)
					]
					this_validation_loss = numpy.mean(validation_losses)
					print(
						'epoch %i, minibatch %i/%i, validation error %f %%' %
						(
							epoch,
							minibatch_index + 1,
							n_train_batches,
							this_validation_loss * 100
						)
					)

					if this_validation_loss < best_validation_loss:
						if this_validation_loss < best_validation_loss * improvement_threshold:
							patience = max(patience, iter * patience_increase)
						best_validation_loss = this_validation_loss
						best_iter = iter
						test_losses = [
							test_model(i) for i in xrange(n_test_batches)
						]
						test_score = numpy.mean(test_losses)

						print(
							'epoch %i, minibatch %i/%i, validation error %f %%' %
							(
								epoch,
								minibatch_index + 1,
								n_train_batches,
								test_score * 100
							)
						)

				if patience <= iter:
					done_looping = True
					break
				
			end_time = timeit.default_timer()
			
			print(
				(
					'Optimization complete with best validation score of %f %%,'
					'with test performance %f %%'
				)
				% (best_validation_loss * 100., test_score * 100.)
			)

			print("the code run for %d epochs, with %f epochs/sec" % (
				epoch, 1. * epoch / (end_time - start_time)
			))

