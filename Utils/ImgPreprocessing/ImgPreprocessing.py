import scipy.ndimage as ndimage
import scipy.misc as misc

import numpy as np

import theano
from theano import function
from theano import tensor as T
from theano.tensor import nlinalg

class PreProcessing:
	def __init__(self, size, patch_size, sigma, resolution):
		if 0 > resolution > (size / 2 - patch_size / 2):
			raise Error("Resolution must be a whole number greater than 0 but less (img_size / 2 - patch_size / 2).")

		self.size = size
		self.resolution = resolution
		self.sigma = sigma
		self.patch_size = patch_size
		self.norm = self.__build_center()

		self.idx = [x for x in range(int((self.size / 2) - (self.patch_size / 2))) if x % self.resolution == 0]
		self.stride = len(self.idx)

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

	def __build_eigens_t3(self):
		t3_covs = T.dtensor3('covs')
		
		# Get eigenvectors
		egmatrix_t3, _ = theano.scan(
			fn=lambda covariance: T.nlinalg.eig(covariance)[1],
			sequences=[t3_covs]
		)
		
		# Transpose is equal to the inverse
		egmatrix_inv_t3, _ = theano.scan(
			fn=lambda egv: egv.dimshuffle(1,0),
			sequences=[egmatrix_t3]
		)

		# Get eigenvalues
		egvalues_t2, _ = theano.scan(
			fn=lambda covariance: T.nlinalg.eig(covariance)[0],
			sequences=[t3_covs]
		)

		# We can't have zero when computing sqrt's because
		# it'll messup theano
		egv_diag, _ = theano.scan(
			fn=lambda x: T.nlinalg.diag(T.maximum(x,.000001)),
			sequences=[egvalues_t2]
		)

		egvalues_t3, _ = theano.scan(
			fn=lambda x: T.nlinalg.matrix_inverse(T.sqrt(x)),
			sequences=[egv_diag]
		)

		return function(inputs=[t3_covs],outputs=[egmatrix_t3, egvalues_t3, egmatrix_t3])

	def __build_eigenvalue_diag(self):

		egvalue_v = T.dvector('egv')

		# Get identity matrix
		identity = T.dmatrix('identity')

		diagonal_root_t2, update3 = theano.scan(
			fn=lambda row, egvalues: row * egvalues,
			sequences=[identity],
			non_sequences=[egvalue_v]
		)

		return function(inputs=[identity,egvalue_v], outputs=[diagonal_root_t2])


	def __build_covariant_loop(self):
		df = T.dvector('f')
		covariant_loop, updates = theano.scan(
			fn=lambda x, row: row * x,
			sequences=[df],
			non_sequences=[df]
		)

		return function(inputs=[df],outputs=[covariant_loop])

	def __build_center(self):
		#We only want to compile our theano functions once
		imgv = T.dtensor3('imgv')
		# Get the mean
		u = T.mean(imgv,0)
		# Get the standard deviation
		s = T.std(T.std(imgv,0),0)
		# Subtract our mean
		return function(inputs=[imgv], outputs=[(imgv - u)/s])

	def __build_zca(self):
		imgv = T.dmatrix('imgv')
		# Diagnol of eigen values
		D = T.dmatrix('imgv')
		# Orthoganol eigen matrix
		E = T.dmatrix('imgv')

	# i is row j is column
	def map_patch_idx(self,img_idx, patch_idx, i, j):
		stride = self.stride
		idx = self.idx
		return img_idx*stride*stride*4  + (i * stride * 4)/ self.resolution + (j * 4)/ self.resolution + patch_idx

	def get_patch(self, imgv, patch_tensor, img_idx):
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
				
	def get_patches(self, t3):
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
			self.get_patch(imgv,patches,idx)
			idx += 1

		return patches

	# Take in imgs and patches and 
	def tmethod_patches(self, patches, tmethod):
		number_of_imgs = patches.shape[0] / (self.stride * self.stride * 4)
		res_patches = np.zeros(
			(
				self.stride * self.stride * 4,
				8,
				8
			),
			dtype=theano.config.floatX
		)

		# used for single filter mean
		sub_patch = np.zeros(
			(
				number_of_imgs,
				8,
				8
			),
			dtype=theano.config.floatX
		)

		for i in self.idx:
			for j in self.idx:
				for sb_idx in range(4):
					for img_idx in range(int(number_of_imgs)):
						sub_patch = patches[self.map_patch_idx(img_idx,sb_idx,i,j)]
					sb = theano.shared(
						value=sub_patch,
						name='sb',
						borrow=True
					)
					res_patches[self.map_patch_idx(0,sb_idx,i,j)] = tmethod(sb, axis=0).eval()

		return res_patches
				 

	def scale(self, imgv):
		# We want to shorten by the largest dimmension
		resize = self.size / float(imgv.shape[0]) if imgv.shape[0] > imgv.shape[1] else self.size / float(imgv.shape[1]) 
		# We then resize the image
		imresize = misc.imresize(imgv, resize, 'bilinear', 'P')
		# The next step is to super impose this onto some background
		# Or image, however we don't care to do that yet
		return imresize
	
	#imgv is a theano sympy
	def normalize(self, patches, mean, std):
		number_of_imgs = patches.shape[0] / (self.stride * self.stride * 4)
		map_patch_idx = self.map_patch_idx
		for img_idx in range(int(number_of_imgs)):
			for i in self.idx:
				for j in self.idx:
					# Get u patches
					u0 = mean[map_patch_idx(0, 0, i, j)]
					u1 = mean[map_patch_idx(0, 1, i, j)]
					u2 = mean[map_patch_idx(0, 2, i, j)]
					u3 = mean[map_patch_idx(0, 3, i, j)]
					
					# Get std patches
					std0 = std[map_patch_idx(0, 0, i, j)]
					std1 = std[map_patch_idx(0, 1, i, j)]
					std2 = std[map_patch_idx(0, 2, i, j)]
					std3 = std[map_patch_idx(0, 3, i, j)]

					patches[map_patch_idx(img_idx, 0, i, j)] = (
						(patches[map_patch_idx(img_idx, 0, i, j)] - u0) / std0
					)
					
					patches[map_patch_idx(img_idx, 1, i, j)] = (
						(patches[map_patch_idx(img_idx, 0, i, j)] - u0) / std0
					)
					
					patches[map_patch_idx(img_idx, 2, i, j)] = (
						(patches[map_patch_idx(img_idx, 0, i, j)] - u0) / std0
					)

					patches[map_patch_idx(img_idx, 3, i, j)] = (
						(patches[map_patch_idx(img_idx, 0, i, j)] - u0) / std0
					)
		return patches

	def spectral_matrices(self, covariances):
		evect_t3, evals_t3, einv_t3 = self.eigens(covariances)
		x = theano.shared(value=evect_t3)
		y = theano.shared(value=evals_t3)
		z = theano.shared(value=einv_t3)
		return theano.scan(
			fn=lambda x, y, z: x.dot(y).dot(z),
			sequences=[x,y,z]
		)[0]


	def covariance(self, sps, u):
		p = theano.shared(
			value=sps,
			name='p',
			borrow=True
		)

		u = theano.shared(
			value=u,
			name='u',
			borrow=True
		)
			
		# (X - u)
		dp = p - u
		dp = dp.reshape((sps.shape[0], self.patch_size ** 2))

		outer = theano.scan(
			fn=lambda x: T.outer(x,x),
			sequences=[dp]
		)

		return T.mean(outer[0], axis=0)

	# Returns whitened patches 1-d vectors
	def whiten(self, spectral_matrices, norms):
		number_of_imgs = norms.shape[0] / (self.stride * self.stride * 4)
		spectral_matrices = spectral_matrices.eval()
		norms_scan = np.zeros(
			(
			norms.shape[0],
			norms.shape[1] ** 2
			),
			dtype=theano.config.floatX
		)
		specs_scan = np.zeros(
			(
				(4 * number_of_imgs * self.stride **2),
				spectral_matrices.shape[1],
				spectral_matrices.shape[2]
			),
			dtype=theano.config.floatX
		)
		inc=0
		for i in self.idx:
			for j in self.idx:
				for img in range(int(number_of_imgs)):
					for q in range(4):
						norms_scan[inc] = norms[self.map_patch_idx(img, q, i, j)].reshape((self.patch_size ** 2))
						specs_scan[inc] = spectral_matrices[self.map_patch_idx(0,q,i,j)]
						inc += 1
		norms_scan = theano.shared(
			value=norms_scan,
			name='ns',
			borrow=True
		)

		specs_scan = theano.shared(
			value=specs_scan,
			name='ss',
			borrow=True
		)

		return theano.scan(
			fn=lambda x, y: x.dot(y),
			sequences=[specs_scan, norms_scan]
		)[0]

	def get_covariance_subs(self, patches, mean):
		number_of_imgs = patches.shape[0] / (self.stride * self.stride * 4)
		sub_patches = []
		u = None
		covariance_subs = np.zeros(
			(
				4 * (self.stride ** 2),
				self.patch_size ** 2,
				self.patch_size ** 2
			),
			dtype=theano.config.floatX
		)
		# used for single filter mean
		sub_patches = np.zeros(
			(
				number_of_imgs,
				8,
				8
			),
			dtype=theano.config.floatX
		)

		for i in self.idx:
			for j in self.idx:
				for sb in range(4):
					for img_idx in range(int(number_of_imgs)):
						sub_patches[img_idx] = patches[self.map_patch_idx(img_idx, 0, i, j)]
						u = mean[self.map_patch_idx(0,sb,i,j)]
					# Now we can calculate our covariance
					covariance_subs[self.map_patch_idx(0,sb,i,j)] = self.covariance(sub_patches, u).eval()
		return covariance_subs
	# This applies a gaussian blur on our image
	def blur(self, imgv):
		return ndimage.gaussian_filter(imgv, self.sigma)