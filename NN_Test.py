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
nf = NOFont.NOFont(training, validation)
#char_classes = list(string.ascii_letters) + [str(i) for i in range(10)]
char_classes = [str(i) for i in range(10)]
print(char_classes)
classes_size = len(char_classes)
char_set_nh = [nh.get_characterset(char) for char in char_classes]
char_set_nf = [nf.get_characterset(char) for char in char_classes]
dataset = []
for char_class_nh, char_class_nf in zip(char_set_nh, char_set_nf):
	ds_nh = char_class_nh.get_dataset(out_len=10);
	ds_nf = char_class_nf.get_dataset(out_len=10);
	# training, validiation, testing
	for type in range(3):
		ds_nh_type = ds_nh[type]
		ds_nh_input = ds_nh_type[0]
		ds_nh_target = ds_nh_type[1]
		
		ds_nf_type = ds_nf[type]
		ds_nf_input = ds_nf_type[0]
		ds_nf_target = ds_nf_type[1]
		
		# Just assign the first array if we do not have
		# anything in our dataset
		if len(dataset) - 1 < type:
			dataset.append([ds_nh_input,ds_nh_target])
			dataset[type][0] = np.concatenate(
				(
					dataset[type][0],
					ds_nf_input
				),
				axis=0
			)
			dataset[type][1] = np.concatenate(
				(
					dataset[type][1],
					ds_nf_target
				),
				axis=0
			)
			continue

		# Concate inputs
		dataset[type][0] = np.concatenate(
			(
				dataset[type][0],
				ds_nf_input
			),
			axis=0
		)

		dataset[type][0] = np.concatenate(
			(
				dataset[type][0],
				ds_nh_input
			),
			axis=0
		)


		dataset[type][1] = np.concatenate(
			(
				dataset[type][1],
				ds_nf_target
			),
			axis=0
		)

		dataset[type][1] = np.concatenate(
			(
				dataset[type][1],
				ds_nh_target
			),
			axis=0
		)
	print(char_class_nh.character)

# Now we need to randomize our array indices
for type in range(3):
	rng = np.random.RandomState()
	state = rng.get_state()
	rng.shuffle(dataset[type][0])
	rng.set_state(state)
	rng.shuffle(dataset[type][1])

# Lets varify that our targets are correct
# for type in range(3):
# 	for i in range(dataset[type][1].shape[0]):
# 		print(dataset[type][0][i].astype(np.uint8))
# 		misc.imsave(
# 			'images/{0}-{1}-{2}.png'.format(
# 				type,i,np.argmax(dataset[type][1][i])
# 				),
# 			dataset[type][0][i].astype(np.uint8)
# 		)

nn = NN(
	# For conv nets we take
	# (None, 1, height, width)
	batch_size=batch_size,
	input_shape=(32, 32),
	# We want to classify 62 characters
	n_out=10
)

# Add our convolution
# Which takes the parameters
# n_kerns, height, and width
# nn.add(
# 	'Convolution',
# 	n_kerns=115,
# 	height=12,
# 	width=12
# )

# # Now we want to add pooling
# nn.add(
# 	'Pool',
# 	shape=(2,2)
# )

# # Add convolution
# nn.add(
# 	'Convolution',
# 	n_kerns=256,
# 	height=5,
# 	width=5
#)

# Add fc layer
nn.add(
	'Convolution',
	n_kerns=115,
	height=12,
	width=12
)

nn.add(
	'Pool',
	shape=(2,2)
)

nn.add(
	'FCLayer',
	n_out=500
)

nn.compile()
nn.set_ttv_data(dataset)
nn.train()




