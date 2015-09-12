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

a = NOFont.NOFont(.1,.4)
batch_size = 50

csetA = a.get_characterset('A')

img_size = 32

print("Get images")
t3_images = csetA.t3_training()

# Reshape our image so it fits our conv net
t3_images = t3_images.reshape((t3_images.shape[0], 1, 32, 32))


batch_size = 50

nn = NN(
	# For conv nets we take
	# (None, 1, height, width)
	input_shape=(batch_size, 1, 32, 32),
	# We want to classify 62 characters
	n_out=62
)

# Add our convolution
# Which takes the parameters
# n_kerns, height, and width
nn.add(
	'Convolution',
	n_kerns=115,
	height=8,
	width=8
)

# Now we want to add pooling
nn.add(
	'Pool',
	shape=(2,2)
)

# Add convolution
nn.add(
	'Convolution',
	n_kerns=20,
	height=4,
	width=4
)

# Add pooling layer
nn.add(
	'Pool',
	shape=(2,2)
)

# Add fc layer
nn.add(
	'FCLayer',
	n_out=500
)

nn.compile()
print(t3_images[0:50].shape)
print(nn.softmax_classify(t3_images[0:50]))