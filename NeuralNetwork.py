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

import dev_logger

import math

import numpy as np

import timeit


# Assume we only classify fonts
class NeuralNetwork(object):
	from enum import Enum
	class DataSet(Enum):
		training = 0
		validation = 1
		testing = 2

	def __init__(self,
		input_shape,
		n_out,
		batch_size,
		preprocessor=None
		):
		self.layers = []
		self.params = []
		self.batch_size = batch_size
		self.input_shape = input_shape
		self.n_out = n_out
		self.output = None
		self.finalized = False
		self.preprocessor = preprocessor
		self.logger = dev_logger.logger(__name__ + ".NeuralNetwork")

	def build_fclayer(self, layer, previous_layer, last_output, **kwargs):
		# We need to reshape the last_output
		# depending on what type of layer we had
		if previous_layer['name'] in ['Pool', 'Convolution']:
			os = previous_layer['output_shape']
			last_output = last_output.reshape((
				os[0],
				os[1] * os[2] * os[3]
			))
			kwargs['n_in'] = os[1] * os[2] * os[3]
		else:
			# We can assume we have an fc layer
			kwargs['n_in'] = previous_layer['output_shape'][1]

		# With FCLayer we can just passing kwargs
		entity = FCLayer(**kwargs)
		layer['output_shape'] = entity.output_shape()

		### Logging ###
		self.logger.info("fclayer output shape")
		self.logger.info(layer['output_shape'])

		return (layer, entity, last_output)

	def build_convolution(self, layer, previous_layer, last_output, **kwargs):
		input_shape = previous_layer['output_shape']
		layer['filter_shape'] = (
			kwargs['n_kerns'],
			input_shape[1],
			kwargs['height'],
			kwargs['width']
		)

		entity = Convolution.withoutFilters(
			filter_shape=layer['filter_shape'],
			image_shape=input_shape
		)
		
		layer['output_shape'] = entity.output_shape()
		layer['image_shape'] = input_shape
		layer['filter_shape'] = layer['filter_shape']

		### Logging ###
		self.logger.info("conv output shape")
		self.logger.info(layer['output_shape'])
		self.logger.info("conv image shape")
		self.logger.info(layer['image_shape'])
		self.logger.info("conv filter shape")
		self.logger.info(layer['filter_shape'])

		return (layer, entity)

	def build_pool(self, layer, previous_layer, last_output, **kwargs):
		entity = Pool(kwargs['shape'])
		if previous_layer['name'] == 'Convolution':
			layer['output_shape'] = entity.output_shape(
				previous_layer['output_shape']
			)

		else:
			layer['output_shape'] = (
				previous_layer['output_shape'] / kwargs['shape']
			)

		### Logging ###
		self.logger.info("pool output shape")
		self.logger.info(layer['output_shape'])

		return (layer, entity)

	# We assume all inputs/ outputs are t3
	def add(self, name, **kwargs):
		if self.finalized:
			raise Exception("You must reset the nerual net before adding layers")

		layer = {
			'name': name
		}

		last_output = None
		previous_layer = None

		if len(self.layers) == 0:
			if layer['name'] == 'Convolution':
				self.inputs = T.ftensor4('inputs')
				last_output = self.inputs
				previous_layer = {
					'output_shape': (self.batch_size, 1,) + self.input_shape
				}
			else:
				self.inputs = T.fmatrix('inputs')
				last_output = self.inputs
		else:
			previous_layer = self.layers[-1]
			last_output = previous_layer['outputs']
		
		if name == 'FCLayer':
			layer, entity, last_output = self.build_fclayer(layer, previous_layer, last_output, **kwargs)
			self.params += entity.params
		
		# Get the filter shape that we need for conv nets
		elif name == 'Convolution':
			layer, entity = self.build_convolution(layer, previous_layer, last_output, **kwargs)
			self.params += entity.params

		elif name == 'Pool':
			layer, entity = self.build_pool(layer, previous_layer, last_output, **kwargs)

		layer['outputs'] = entity.get_outputs(last_output)
		layer['entity'] = entity
		self.layers.append(layer)

	# This will transform our inputs before it's fed
	# into the net
	def set_preprocessor(self, preprocessor):
		self.preprocessor = preprocessor

	# We could add the option of choosing the end classifier,
	# but softmax is the easiest way to normalize and
	# binary_crossentropy seems like the best method that I
	# know of.
	def compile(self):
		# get our output
		self.add('FCLayer', n_out=self.n_out, activation=T.nnet.softmax)

		# We should crash and burn if someone trys to compile
		# without any layers.
		if len(self.layers) == 0:
			raise Exception("Cannot compile a neural network without layers")

		# finalize our neural net so we can't add anymore layers
		self.finalized = True

		output = self.layers[-1]['outputs']
		
		# Define our input
		self.softmax_classify_fn = theano.function(
			inputs=[self.inputs],
			outputs=[output]
		)

	@staticmethod
	def pad_with_zeros(input, batch_size):
		padding = (
			((math.ceil(input.shape[0] / batch_size) * \
			batch_size)) - input.shape[0]
		)

		padding_tensor = np.zeros((padding,) + input.shape[1:])

		return np.vstack((input, padding_tensor))

	@staticmethod
	def pad_with_wrap(input, batch_size):
		batch_size

		padding = (
			((math.ceil(input.shape[0] / batch_size) * \
			batch_size)) - input.shape[0]
		)

		padding_tensor = None

		## This loop gives us the number of times we need to wrap
		for wrap_index in range(1, math.ceil(padding / input.shape[0]) + 1):
			### if input > padding ###
			if padding - input.shape[0] < 1:
				padding_tensor = input[0:padding]
			elif wrap_index * input.shape[0] < padding:
				if type(padding_tensor).__module__ != np.__name__:
					padding_tensor = input
				else:
					padding_tensor = np.vstack((
						padding_tensor,
						input
					))
			else:
				padding_tensor = np.vstack((
					padding_tensor,
					input[0: padding - wrap_index * input.shape[0]]
				))

		if type(padding_tensor).__module__ != np.__name__:
			return input
		else:
			return np.vstack((input,padding_tensor))

	# Returns a tuple of: (argmax(softmax), (sm0, sm1, ... smN))
	def softmax_classify(self, input):
		results = []

		if self.preprocessor != None:
			input = self.preprocessor(input)

		input = NeuralNetwork.pad_with_zeros(input, self.batch_size).astype(theano.config.floatX)

		if self.layers[0]['name'] == 'Convolution':
			input = input.reshape((input.shape[0], 1,) + self.input_shape)

		for index in range(math.ceil(input.shape[0]/self.batch_size)):
			index_start = index * self.batch_size
			index_stop = (index + 1) * self.batch_size
			batch = input[index_start : index_stop]
			results.append(self.softmax_classify_fn(batch))

		return results
	
	# Set our training, testing, and validation data
	# Where data is organized as [(input, target)]
	# Our data set is simply non shared data. We convert
	# Regular data to shared.
	
	# data_set = [(inputs, targets),(inputs, target),(inputs, target)]
	
	def set_ttv_data(self, data_set):

		inputs = []
		targets = []

		for index in range(len(data_set)):
			data_set_pair = data_set[index]
			
			## Wrap our data_sets ##
			inputs.append(NeuralNetwork.pad_with_wrap(
				data_set_pair[0],
				self.batch_size
			).astype(theano.config.floatX))
			targets.append(NeuralNetwork.pad_with_wrap(
				data_set_pair[1],
				self.batch_size
			).astype(theano.config.floatX))

			## Reshape our data_sets if necessary ##
			if self.layers[0]['name'] == 'Convolution':
				inputs[-1] = inputs[-1].reshape((inputs[-1].shape[0], 1,) + self.input_shape)

			## Build our shared data_sets ##
			inputs[-1] = theano.shared(inputs[-1], borrow=True)
			targets[-1] = theano.shared(targets[-1], borrow=True)

		self.inputs_ds = inputs
		self.targets_ds = targets

	def train(self,
		learning_rate=.0001,
		l2_rate=.0001,
		l1_rate=.0001,
		patience=10000,
		patience_increase=2,
		improvement_threshold=0.995
		):
		targets = T.fmatrix('targets')
		batch_size = self.batch_size
		outputs = self.layers[-1]['outputs']

		params = self.params

		w_list = [T.sum(T.abs_(param[0].flatten())) for param in params]
		w_sum = sum(w_list)
		
		w_sqr_list = [T.sum(T.sqr(param[0].flatten())) for param in params]
		w_sqr_sum = sum(w_sqr_list)

		w_num_list = [param[0].flatten().shape[0] for param in params]
		w_num = sum(w_num_list)
		
		l2 = w_sqr_sum * l2_rate

		l1 = w_sum * l1_rate

		# Build out our training model
		cost = T.mean(
			T.nnet.binary_crossentropy(
				outputs, targets
			) + l1 + l2
		)
		
		
		grads = T.grad(cost, params)

		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		index = T.lscalar()
		input_training_sh = self.inputs_ds[NeuralNetwork.DataSet.training.value]
		target_training_sh = self.targets_ds[NeuralNetwork.DataSet.training.value]
		self.training_model = theano.function(
			inputs=[index],
			outputs=[cost],
			updates=updates,
			givens={
				self.inputs: input_training_sh[index * batch_size: (index + 1) * batch_size],
				targets: target_training_sh[index * batch_size: (index + 1) * batch_size]
			}
		)

		cnn_out = T.argmax(self.layers[-1]['outputs'],axis=1)

		# Shared testing and validation targets
		target_testing_sh = self.targets_ds[NeuralNetwork.DataSet.testing.value]
		target_validation_sh = self.targets_ds[NeuralNetwork.DataSet.validation.value]

		# Shared testing and validation inputs
		input_testing_sh = self.inputs_ds[NeuralNetwork.DataSet.testing.value]
		input_validation_sh = self.inputs_ds[NeuralNetwork.DataSet.validation.value]

		# Collapse our matrix into a vector
		target_testing_sh = T.argmax(target_testing_sh,axis=1)
		target_validation_sh = T.argmax(target_validation_sh,axis=1)

		# Declare our target outs
		target_testing = T.lvector("target_testing")
		target_validation = T.lvector("target_validation")

		# Create our error tests
		error_testing = T.mean(T.neq(target_testing, cnn_out))
		error_validation = T.mean(T.neq(target_validation, cnn_out))

		testing_model = theano.function(
			[index],
			outputs=[error_testing],
			givens={
				self.inputs: input_testing_sh[index * batch_size: (index + 1) * batch_size],
				target_testing: target_testing_sh[index * batch_size: (index + 1) * batch_size]
			}
		)

		validate_model = theano.function(
			[index],
			outputs=[error_validation],
			givens={
				self.inputs: input_validation_sh[index * batch_size: (index + 1) * batch_size],
				target_validation: target_validation_sh[index * batch_size: (index + 1) * batch_size]
			}
		)

		self.logger.info("Training!")

		self.logger.info(testing_model(0))
		iterations = 0
		error_model_old = 2
		error_model_new = testing_model(0)[0]
		epsilon = 0.001
		training_range = int(target_training_sh.shape[0].eval() / 50)
		testing_range = int(target_testing_sh.shape[0].eval() / 50)
		print(training_range)
		print(testing_range)
		while True:
				for i in range(training_range):
					self.logger.info(self.training_model(i))
					error_model_list = [ testing_model(i)[0] for i in range(testing_range) ]
					error_model_old = error_model_new
					error_model_new = sum(error_model_list)/testing_range
					print(error_model_new)

	# def train(self,
	# 	patience=10000,
	# 	patience_increase=2,
	# 	improvement_threshold=0.995
	# ):
	# 	print("... training the model")

	# 	validation_frequency = min(n_train_batches, patience / 2)

	# 	best_validation_loss = np.inf
	# 	best_iter = 0
	# 	test_score = 0.
	# 	start_time = timeit.default_timer()

	# 	done_looping = False
	# 	epoch = 0
	# 	while (epoch < n_epochs) and (not done_looping):
	# 		epoch = epoch + 1
	# 		for minibatch_index in range(n_train_batches):
	# 			minbatch_avg_cost = train_model(minibatch_index)
	# 			iter = (epoch - 1) * n_train_batches + minibatch_index
	# 			if (iter + 1) % validation_frequency == 0:
	# 				validation_losses = [
	# 					validate_model(i) for i in xrange(n_valid_batches)
	# 				]
	# 				this_validation_loss = numpy.mean(validation_losses)
	# 				print(
	# 					'epoch %i, minibatch %i/%i, validation error %f %%' %
	# 					(
	# 						epoch,
	# 						minibatch_index + 1,
	# 						n_train_batches,
	# 						this_validation_loss * 100
	# 					)
	# 				)

	# 				if this_validation_loss < best_validation_loss:
	# 					if this_validation_loss < best_validation_loss * improvement_threshold:
	# 						patience = max(patience, iter * patience_increase)
	# 					best_validation_loss = this_validation_loss
	# 					best_iter = iter
	# 					test_losses = [
	# 						test_model(i) for i in xrange(n_test_batches)
	# 					]
	# 					test_score = numpy.mean(test_losses)

	# 					print(
	# 						'epoch %i, minibatch %i/%i, validation error %f %%' %
	# 						(
	# 							epoch,
	# 							minibatch_index + 1,
	# 							n_train_batches,
	# 							test_score * 100
	# 						)
	# 					)

	# 			if patience <= iter:
	# 				done_looping = True
	# 				break
				
	# 		end_time = timeit.default_timer()
			
	# 		print(
	# 			(
	# 				'Optimization complete with best validation score of %f %%,'
	# 				'with test performance %f %%'
	# 			)
	# 			% (best_validation_loss * 100., test_score * 100.)
	# 		)

	# 		print("the code run for %d epochs, with %f epochs/sec" % (
	# 			epoch, 1. * epoch / (end_time - start_time)
	# 		))

