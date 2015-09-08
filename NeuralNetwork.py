import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessing
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

import Learning.Supervised as supervised_learning
import Learning.Unsupervised as usupervised_learning
import Learning.Costs as costs

import numpy as np

import timeit

# Assume we only classify fonts
class NeuralNetwork(object):
	def __init__(self,
		training,
		validation,
		batch_size=600,
		img_shape=(32,32),
		patch_size=8,
		sigma=1.2,
		resolution=4,
		k_filters=115,
		k_iterations=10):
		self.batch_size
		self.training = training
		self.validation = validation
		self.nh = NOHandwritting.NOHandwritting(training, validation)
		char_classes = list(string.letters) + list(range(10))
		self.classes_size = len(char_classes)
		self.char_set = [self.nh.get_characterset(char) for char in char_classes]

		# Image preprocessor will transform the image before sends it as input into our
		# neural net
		self.pp = ImgPP(size=img_shape, patch_size=patch_size, sigma=sigma, resolution=4)
		self.ks = learning_unsupervised.KSphere(k_filters, k_iterations)

	def build_filters(self):
		return None

	def zca_whiten(self):
		return None

	# We assume all inputs/ outputs are t3
	def add(self, layer):
		self.layers[-1].get_output(self.output)

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

# t3 data blurred
print("Building data set")
t3 = csetA.t3_training(pp)
t3_shared = theano.shared(
	value=np.asarray(
          t3,
          dtype=theano.config.floatX
    ),
    borrow=True
)

print("Retrieving patches")
patches = pp.get_patches(t3)
print("Building mean patches")
mean_patches = pp.tmethod_patches(patches, T.mean)
print("Building std patches")
std_patches = pp.tmethod_patches(patches, T.std)
s = theano.shared(
	value=std_patches,
	name='s',
	borrow=True
)
std_patches = T.maximum(s,.0000001).eval()

print("Normalizing patches")
normalized_patches = pp.normalize(patches,mean_patches,std_patches)

print("Get covariant matrices")
covariant = pp.get_covariance_subs(normalized_patches, mean_patches)

print("Retrieving spectral matrices")
spectral_matrices = pp.spectral_matrices(covariant)

print("Applying whitening and centering")
whitened_patches = pp.whiten(spectral_matrices, normalized_patches)

print("Whitening successful! Lets do some unsuppervised learning!!!")
# Returns matrix of 2d matrix
D_Filters = ks_115.spherical_k(whitened_patches)

# x is our input average
x = T.matrix('x')
y = T.ivector('y')

print("Convolve0")
conv0 = l_s.Convolution.withFilters(
	image_shape=t3.shape,
	filters=D_Filters.reshape((96,8,8))
)
feature_maps0 = conv0.get_output(x)

print("Pool0")
pool0 = l_s.Pool((2,2))
pool_out0 = pool0.get_output(feature_maps0)
pool_out0 = pool_out0.flatten()

print("FCLayer 0")
fc0 = l_s.FCLayer(
	pool_out0.shape[0].eval(),
	500
)

fc0_out = fc0.get_output(pool_out0)

print("Get softmax output")
soft0 = l_s.FCLayer(
	500,
	62,
	activation=T.nnet.softmax
)

output = T.argmax(soft0.get_output(fc0_out))

params = soft0.params + fc0.params + conv0.params

cost = costs.cross_entropy(output)

grads = T.grad(cost(y), params)

updates = [
	(param_i, param_i - .001 * grad_i)
	for param_i, grad_i in zip(params, grads)
]

index = T.lscalar()

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=updates,
    givens={
      x: t3_shared[index * batch_size: (index + 1) * batch_size],
      y: y[index * batch_size: (index + 1) * batch_size]
    }
  )
