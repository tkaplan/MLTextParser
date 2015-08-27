import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessoring
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

import theano
from theano import tensor as T
from theano import function

import Learning.Unsupervised as l_u

import numpy as np

a = NOFont.NOFont(.1,.4)
b = NOHandwritting.NOHandwritting(.01,.4)

assert a.get_classpath('a') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample037'
assert a.get_classpath('9') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample010'
assert a.get_classpath('8') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample009'
assert a.get_classpath('0') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample001'
assert a.get_classpath('A') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample011'

csetA = a.get_characterset('A')
print "Number of training images"
# We instantiate our image preprocessor
pp = ImgPP(size=32, patch_size=8, sigma=1.2, resolution=4)

# KSphere with 96 filters and 10 iterations
ks_96 = l_u.KSphere(96, 10)

# t3 data blurred
print "Building data set"
t3 = csetA.t3_training(pp)
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
ks_96.spherical_k(whitened_patches)
# variance_patches = pp.variance(patches, mean_patches)
# centered_patches = pp.center(mean_patches)
# whitened_filters = pp.whiten(cenetered_patches)

# Now we are ready to run our k-means regression
# on our whitened filters


# imgv = csetA.next_training()
# imgv = pp.scale(imgv)
# imgv = pp.blur(imgv)
# print imgv.shape
# print pp.get_patch(imgv)
#print t3.shape