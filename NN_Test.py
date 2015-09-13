from NeuralNetwork import NeuralNetwork as NN



import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

from Utils.PreProcessing import ZCA_Whitening, Patch

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

from Learning.Supervised import Convolution, Pool, FCLayer
from Learning.Unsupervised import KSphere
import Learning.Costs as costs

import numpy as np
import string

batch_size = 50
training = .5
validation = .25

nh = NOHandwritting.NOHandwritting(training, validation)
char_classes = list(string.ascii_letters) + [str(i) for i in range(10)]
print(char_classes)
classes_size = len(char_classes)
char_set = [nh.get_characterset(char) for char in char_classes]

# for char_class in char_set:
	
# 	# setup training set
# 	training = char_class.t3_training()
# 	training_input_set.append(training)
# 	training_target_set.append(char_class.get_target_vector)
	
# 	# setup testing set
# 	testing = char_class.t3_testing()
# 	testing_input_set.append(testing)
# 	testing_target_set.append(char_class.get_target_vector)

# 	# setup validation set
# 	validation = char_class.t3_validation()
# 	validation_input_set.append(validation)
# 	validation_target_set.append(char_class.target_vector)
# batch_size = 50

# nn = NN(
# 	# For conv nets we take
# 	# (None, 1, height, width)
# 	batch_size=batch_size,
# 	input_shape=(32, 32),
# 	# We want to classify 62 characters
# 	n_out=62
# )

# # Add our convolution
# # Which takes the parameters
# # n_kerns, height, and width
# nn.add(
# 	'Convolution',
# 	n_kerns=115,
# 	height=8,
# 	width=8
# )

# # Now we want to add pooling
# nn.add(
# 	'Pool',
# 	shape=(2,2)
# )

# # Add convolution
# nn.add(
# 	'Convolution',
# 	n_kerns=20,
# 	height=4,
# 	width=4
# )

# # Add pooling layer
# nn.add(
# 	'Pool',
# 	shape=(2,2)
# )

# # Add fc layer
# nn.add(
# 	'FCLayer',
# 	n_out=500
# )

# nn.compile()
# print(nn.softmax_classify(t3_images))