import theano
import theano.tensor as T

import numpy as np

def cross_entropy(output):
	def get_cost(y):
		return T.mean(y*T.log(output) + (1-y)*T.log(1-output)[T.arange(y.shape[0]),y])
	return get_cost