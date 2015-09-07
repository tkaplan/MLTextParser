import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessing
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

import Learning.Supervised as l_s
import Learning.Unsupervised as l_u
import Learning.Costs as costs

import numpy as np

a = NOFont.NOFont(.1,.4)
b = NOHandwritting.NOHandwritting(.01,.4)
batch_size = 600

csetA = a.get_characterset('A')

img_shape = (32, 32)

print("Number of training images")
# We instantiate our image preprocessor
pp = ImgPP(size=img_shape[0], patch_size=8, sigma=1.2, resolution=4)

# KSphere with 96 filters and 10 iterations
ks_115 = l_u.KSphere(115, 10)

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

