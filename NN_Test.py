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
#char_classes = list(string.ascii_letters) + [str(i) for i in range(10)]
char_classes = [str(i) for i in range(10)]
print(char_classes)
classes_size = len(char_classes)
char_set = [nh.get_characterset(char) for char in char_classes]
dataset = []
for char_class in char_set:
	ds = char_class.get_dataset();
	# training, validiation, testing
	for type in range(3):
		ds_type = ds[type]
		ds_input = ds_type[0]
		ds_target = ds_type[1]
		
		# Just assign the first array if we do not have
		# anything in our dataset
		if len(dataset) - 1 < type:
			dataset.append([ds_input,ds_target])
			continue

		# Concate inputs
		dataset[type][0] = np.concatenate(
			(
				dataset[type][0],
				ds_input
			),
			axis=0
		)


		dataset[type][1] = np.concatenate(
			(
				dataset[type][1],
				ds_target
			),
			axis=0
		)
	print(char_class.character)

# Now we need to randomize our array indices
for type in range(3):
	np.random.shuffle(dataset[type][0])
	np.random.shuffle(dataset[type][1])

batch_size = 50

nn = NN(
	# For conv nets we take
	# (None, 1, height, width)
	batch_size=batch_size,
	input_shape=(32, 32),
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
nn.set_ttv_data(dataset)
nn.train()




