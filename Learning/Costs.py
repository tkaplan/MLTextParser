import theano
import theano.tensor as T

import numpy as np

def cross_entropy(out):
	def get_cost(y):
		return T.mean(y*T.log(out) + (1-y)*T.log(1-out)[T.arange(y.shape[0]),y])
		#return -T.mean(T.log(out)[T.arange(y.shape[0]), y])
	return get_cost