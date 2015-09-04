import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessoring
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

import Learning.Supervised as l_s
import Learning.Unsupervised as l_u

import numpy as np

a = NOFont.NOFont(.1,.4)
b = NOHandwritting.NOHandwritting(.01,.4)
batch_size = 600

assert a.get_classpath('a') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample037'
assert a.get_classpath('9') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample010'
assert a.get_classpath('8') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample009'
assert a.get_classpath('0') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample001'
assert a.get_classpath('A') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample011'

csetA = a.get_characterset('A')

img_shape = (32, 32)

print "Number of training images"
# We instantiate our image preprocessor
pp = ImgPP(size=img_shape[0], patch_size=8, sigma=1.2, resolution=4)

# KSphere with 96 filters and 10 iterations
ks_96 = l_u.KSphere(96, 10)

# t3 data blurred
print "Building data set"
t3 = csetA.t3_training(pp)
t3_shared = theano.shared(
	value=np.asarray(
          t3,
          dtype=theano.config.floatX
    ),
    borrow=True
)

print "Retrieving patches"
patches = pp.get_patches(t3)
print "Building mean patches"
mean_patches = pp.tmethod_patches(patches, T.mean)
print "Building std patches"
std_patches = pp.tmethod_patches(patches, T.std)
s = theano.shared(
	value=std_patches,
	name='s',
	borrow=True
)
std_patches = T.maximum(s,.0000001).eval()

print "Normalizing patches"
normalized_patches = pp.normalize(patches,mean_patches,std_patches)

print "Get covariant matrices"
covariant = pp.get_covariance_subs(normalized_patches, mean_patches)

print "Retrieving spectral matrices"
spectral_matrices = pp.spectral_matrices(covariant)

print "Applying whitening and centering"
whitened_patches = pp.whiten(spectral_matrices, normalized_patches)

print "Whitening successful! Lets do some unsuppervised learning!!!"
# Returns matrix of 2d matrix
D_Filters = ks_96.spherical_k(whitened_patches)

print "Convolve0"
conv0 = l_s.Convolution.withFilters(
	image_shape=t3.shape,
	filters=D_Filters.reshape((96,8,8))
)
feature_maps0 = conv0.get_output(t3_shared)

print "Pool0"
pool0 = l_s.Pool((2,2))
pool_out0 = pool0.get_output(feature_maps0)
pool_out0 = pool_out0.flatten()

# print "Convolve1"
# conv1 = l_s.Convolution.withoutFilters(
# 	image_shape=tuple(pool_out0.shape.eval()),
# 	filter_shape=(pool_out0.shape[0].eval(),5,4,4)
# )
# feature_maps1 = conv1.get_output(pool_out0)

# print "Pool1"
# pool1 = l_s.Pool((2,2))
# pool_out1 = pool1.get_output(feature_maps1)
# fc_input = pool_out1.flatten()

# print fc_input.shape.eval()
# print fc_input.eval()

# Convert pool_out0 to single array output, this will feed
# directly into our binary softmax classifier
print "FCLayer 0"
fc0 = l_s.FCLayer(
	pool_out0.shape[0].eval(),
	500
)

fc0_out = fc0.get_output(pool_out0)

print "Get softmax output"
soft0 = l_s.FCLayer(
	500,
	2,
	activation=T.nnet.softmax
)

output = soft0.get_output(fc0_out)

print output.eval()

