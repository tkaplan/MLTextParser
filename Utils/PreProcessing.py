import scipy.ndimage as ndimage
import scipy.misc as misc

import numpy as np

import theano
from theano import function
from theano import tensor as T
from theano.tensor import nlinalg

import dev_logger

class ZCA_Whitening:
	def __init__(self):
		np.set_printoptions(threshold=np.nan)
		# Set up file handler for file logging
		self.logger = dev_logger.logger(__name__ + ".ZCA_Whitening")

		self.cov_matrix = np.zeros(
			(
				self.stride * 4,
				self.patch_size ** 2,
				self.patch_size ** 2	
			),
			dtype=theano.config.floatX
		)

		self.covariant_loop = self.__build_covariant_loop()

		# Input covariant tensor 3 get eigens tensor 3
		self.eigens = self.__build_eigens_t3()
		# Input identity matrix and eigenvalues
		self.eigen_diag = self.__build_eigenvalue_diag()

	def __spectral_matrix(self, covariance):
		egvalues, egmatrix = T.nlinalg.eig(covariance)
		egmatrix_inv = T.nlinalg.matrix_inverse(egmatrix)
		diag_sqr_inv = T.nlinalg.alloc_diag(
			T.inv(
				T.sqrt(
					T.switch(T.eq(egvalues,0), 0.001, egvalues)
				)
			)
		)
		return egmatrix.dot(diag_sqr_inv).dot(egmatrix_inv)

	def process(self, t3):
		logger.info("Retrieving patches")
		patches = self__get_patches(t3)
		
		logger.info("Reshaping patches")
		patches = patches.reshape((
			t3.shape[0],
			t3.shape[1] * t3.shape[2]
		))
		
		logger.info("Building means")
		u_patch = patches.mean(axis=0)

		logger.info("Building std")
		# Add .001 to s_patch so we don't divide by zero
		s_patch = patches.std(axis=0) + .001

		logger.info("Normalize patches")
		n_patches = (patches - u_patch) / s_patch

		logger.info("Covariant matrix")
		dp = n_patches - u_patch
		# To get the covariance tensor, we find
		# the covariance of each vector with itself
		# and take the mean
		covariance = T.mean(
			theano.scan(
				fn=lambda x: T.outer(x,x),
				sequences=[dp]
			)[0],
			axis=0
		)

		logger.info("Spectral matrices")
		spectrum = self.__spectral_matrix(covariance)

		logger.info("Applying whitening and centering")
		return theano.scan(
			fn=lambda row, spectrum: spectrum.dot(row),
			sequences=[n_patches],
			non_sequences=[spectrum]
		)[0]


class Patch(object):
	def __init__(self, size, patch_size, sigma, resolution):
		np.set_printoptions(threshold=np.nan)
		# Set up file handler for file logging
		self.logger = dev_logger.logger(__name__ + ".Patch")

		if 0 > resolution > (size / 2 - patch_size / 2):
			raise Error("Resolution must be a whole number greater than 0 but less (img_size / 2 - patch_size / 2).")
		self.size = size
		self.resolution = resolution
		self.sigma = sigma
		self.patch_size = patch_size

		self.idx = [x for x in range(int((self.size / 2) - (self.patch_size / 2))) if x % self.resolution == 0]
		self.stride = len(self.idx)

	def __get_patch(self, imgv, patch_tensor, img_idx):
		idx = self.idx
		for i in idx:
			for j in idx:
				x = i
				xc = self.size-i-self.patch_size
				y = j
				yc = self.size-j-self.patch_size

				# Append patch 0 to tensor
				patch_tensor[self.map_patch_idx(img_idx,0,i,j)] = imgv[x:x+self.patch_size,y:y+self.patch_size]		
				patch_tensor[self.map_patch_idx(img_idx,1,i,j)] = imgv[xc:xc+self.patch_size,y:y+self.patch_size]
				patch_tensor[self.map_patch_idx(img_idx,2,i,j)] = imgv[x:x+self.patch_size, yc:yc+self.patch_size]
				patch_tensor[self.map_patch_idx(img_idx,3,i,j)] = imgv[xc:xc+self.patch_size ,yc:yc+self.patch_size]
		return patch_tensor
				
	def __get_patches(self, t3):
		patches = np.empty(
			(
				self.map_patch_idx(1,0,0,0) * t3.shape[0],
				8,
				8
			),
			dtype=theano.config.floatX
		)
		idx = 0
		for imgv in t3:
			self.__get_patch(imgv,patches,idx)
			idx += 1

		return theano.shared(
			value=patches,
			borrow=True
		)

	# i is row j is column
	def map_patch_idx(self,img_idx, patch_idx, i, j):
		stride = self.stride
		idx = self.idx
		return img_idx*stride*stride*4  + (i * stride * 4)/ self.resolution + (j * 4)/ self.resolution + patch_idx