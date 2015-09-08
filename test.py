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
batch_size = 600

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
x = T.matrix('x')
y = T.ivector('y')

print("Convolve0")
conv0 = Convolution.withFilters(
	image_shape=t3_images.shape[0:1] + (1,) + t3_images.shape[-2:],
	filters=filters_multi.reshape((115,1,8,8))
)

t3_images = t3_images.reshape(t3_images.shape[0:1] + (1,) + t3_images.shape[-2:])

fm0 = conv0.get_output(theano.shared(t3_images,borrow=True))

print(fm0.shape.eval())

print("Pool0")
pool0 = Pool((2,2))
pool_out0 = pool0.get_output(fm0)

print(pool_out0.shape.eval())

conv1 = Convolution.withoutFilters(
	filter_shape=(20,115,4,4),
	image_shape=pool_out0.shape.eval()
)

fm1 = conv1.get_output(pool_out0)

pool1 = Pool((2,2))
pool_out1 = pool1.get_output(fm1)

print("######")
print(pool_out1.shape.eval())

pool_out1 = pool_out1.flatten(2)

print(pool_out1.shape.eval())

print("FCLayer 0")
fc0 = FCLayer(
	pool_out1.shape[1].eval(),
	n_out=500
)

fc0_out = fc0.get_output(pool_out1)
print(fc0_out.shape.eval())

print("Get softmax output")
soft0 = FCLayer(
	500,
	62,
	activation=T.nnet.softmax
)

output = T.argmax(soft0.get_output(fc0_out), axis=1)

print(output.eval())

# params = soft0.params + fc0.params + conv0.params

# cost = costs.cross_entropy(output)

# grads = T.grad(cost(y), params)

# updates = [
# 	(param_i, param_i - .001 * grad_i)
# 	for param_i, grad_i in zip(params, grads)
# ]

# index = T.lscalar()

# train_model = theano.function(
#     inputs=[index],
#     outputs=cost,
#     updates=updates,
#     givens={
#       x: t3_shared[index * batch_size: (index + 1) * batch_size],
#       y: y[index * batch_size: (index + 1) * batch_size]
#     }
#   )

