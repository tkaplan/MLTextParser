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

# We instantiate our image preprocessor
print("Get patches")
patch_processor = Patch(size=32, patch_size=8, resolution=4)
t3_patches = patch_processor.get_patches(t3_images)

print("Whiten training patches")
zca = ZCA_Whitening()
whitened_patches = zca.process(t3_patches)

print("Learn filters")
ks_115 = KSphere(115, 10)
ks_95 = KSphere(95, 10)
filters_multi = ks_115.spherical_k(whitened_patches)
filters_bin = ks_95.spherical_k(whitened_patches)

# x is our input average
# We define our x input and expected y out
x = T.tensor4('x', dtype=theano.config.floatX)
y = T.matrix('y', dtype=theano.config.floatX)

t3_images = t3_images.reshape(t3_images.shape[0:1] + (1,) + t3_images.shape[-2:])
t3_images_shared = theano.shared(t3_images, borrow=True)

# Define out
a = np.ones((batch_size,1))
y_a = np.append(a,np.zeros((50,61)), 1)


t3_y_out_shared = theano.shared(
	value=y_a.astype(theano.config.floatX),
	borrow=True
)

filters_multi = theano.shared(
	value=filters_multi.reshape((115,1,8,8)).eval().astype(theano.config.floatX),
	borrow=True
)

print("Convolve0")
print((50,) + (1,) + t3_images.shape[-2:])
nkernels=[115,20]
conv0 = Convolution.withFilters(
	filter_shape=(nkernels[0], 1, 8, 8),
	image_shape=(batch_size, 1,) + t3_images.shape[-2:],
	filters=filters_multi
)

fm0 = conv0.get_output(x)

print("Pool0")
pool0 = Pool((2,2))
pool_out0 = pool0.get_output(fm0)


conv1 = Convolution.withoutFilters(
	filter_shape=(nkernels[1],nkernels[0],4,4),
	image_shape=(batch_size, nkernels[0],  12,  12)
)

fm1 = conv1.get_output(pool_out0)

pool1 = Pool((2,2))
pool_out1 = pool1.get_output(fm1)
pool_out1 = pool_out1.flatten(2)

print("FCLayer 0")
fc0 = FCLayer(
	n_in=320,
	n_out=500
)

fc0_out = fc0.get_output(pool_out1)

print("Get softmax output")
soft0 = FCLayer(
	500,
	62,
	activation=T.nnet.softmax
)

output = soft0.get_output(fc0_out)
params = soft0.params + fc0.params + conv1.params + conv0.params

cost = T.mean(T.nnet.binary_crossentropy(y, output))
grads = T.grad(cost, params)

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
		x: t3_images_shared[index * batch_size: (index + 1) * batch_size],
		y: t3_y_out_shared[index * batch_size: (index + 1) * batch_size]
	}
)

train_model(0)
print("Fucking success!!! :)")