import theano.tensor as T
import theano

import numpy as np

import math

import dev_logger

class KSphere(object):
	def __init__(self, d_size, iterations):
		np.set_printoptions(threshold=np.nan)
		# Set up file handler for file logging
		self.logger = dev_logger.logger(__name__ + "KSPhere")

		self.d_size = d_size
		self.iterations = iterations
		return None

	# Assume m is a shared variable
	def __normalize_matrix(self, m):
		# Notify logging Normalization
		self.logger.info("Normalizing Matrix")

		ncs = T.sqrt(m.dot(m.transpose()).diagonal())

		self.logger.info("L2 Norm")

		self.logger.debug(ncs.eval())

		self.logger.info("Getting normalized D_t2")

		loop, _ = theano.scan(
			fn=lambda row, nc: row / nc,
			sequences=[m, ncs]
		)

		self.logger.info("Normalized D_t2 done")
		
		return loop

	# X is a matrix consisting of all our observations
	# X needs to be a shared varaible
	def __build_dictionary(self, X):
		d_size = self.d_size
		rng = np.random.RandomState()
		vect_size = X.shape[1].eval()
		x_size = X.shape[0].eval()
		S_group_size =  math.ceil((x_size)/(1.0*d_size))

		self.logger.info("Building D shared")

		# First we initialize our dictionary
		D = theano.shared(
			value=rng.uniform(
				low=-1,
				high=1,
				size=(d_size, vect_size)
			),
			name='D',
			borrow=True
		)

		self.logger.info("D.shape[0].eval()")
		#self.logger.debug(D.shape[0].eval())
		self.logger.info("D.shape[1].eval()")
		#self.logger.debug(D.shape[1].eval())
		self.logger.info("D")
		#self.logger.debug(D.eval())
		self.logger.info("Normalizing D to create D_t2")

		# The initial random dictionary is now normalized
		self.D_t2 = self.__normalize_matrix(D)

		self.logger.info("Creating S_t3 shared memory with zeroed out arrays.")

		# It doesn't matter what S intially is
		self.S_t3 = theano.shared(
			value=np.zeros(
				(d_size, S_group_size, vect_size),
				dtype=theano.config.floatX
			),
			name='S',
			borrow=True
		)

	# Observations is a shared variable
	def spherical_k(self, observations):

		self.logger.info("Initializing spherical_k")

		X = observations
		X_np = observations.eval()

		vect_size = X.shape[1].eval()

		self.logger.info("Number of observations")
		self.logger.debug(X.shape[0].eval())

		self.logger.info("Building dictionary and encoding vectors")

		self.__build_dictionary(X)

		self.logger.info("Iterating through dictionary")

		D = np.zeros(
			(self.d_size, vect_size),
			dtype=theano.config.floatX
		)
		
		for it in range(10):
			self.logger.info("##### S Means Iteration {0} ######".format(it))
			
			j_t2 = theano.scan(
				fn=lambda xi, D_t2: T.argmax(D_t2.dot(xi)),
				sequences=[X],
				non_sequences=[self.D_t2]
			)[0].eval()

			for idx in range(self.d_size):
				# Find the mean for each filterIdx
				x_idx = np.where(j_t2 == idx)[0]
				self.logger.info("##### Number for filter {0} #######".format(idx))
				self.logger.info(x_idx.size)
				d_filter = None

				if x_idx.size != 0:
					d_filter = theano.shared(
						value=np.asarray([X_np[i] for i in x_idx]),
						name='d_filter',
						borrow=True
					).mean(axis=0) + .0001
				else:
					d_filter = theano.shared(
						value=np.zeros((vect_size)),
						name='d_filter',
						borrow=True
					) + .0001


				self.logger.debug("##### Observations ####")
				self.logger.debug(d_filter.eval())

				D[idx] = d_filter.eval()

			D_SH = theano.shared(
				value=D,
				name="D",
				borrow=True
			)

			self.D_t2 = self.__normalize_matrix(D_SH)
			self.logger.debug(self.D_t2.eval())
			return self.D_t2

	def opt_s(self, j):
		return None
		# We now want to iterate through our observations
		# to update S_t3

	def opt_D(self, j):
		return None
