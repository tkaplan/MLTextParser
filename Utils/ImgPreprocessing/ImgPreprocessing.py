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

		self.idx = [x for x in range((self.size / 2) - (self.patch_size / 2)) if x % self.resolution == 0]
		self.stride = len(self.idx)

		self.cov_matrix = np.zeros(
			(
				self.stride * 4,
				self.patch_size ** 2,
				self.patch_size ** 2	
			)
		)

		self.covariant_loop = self.__build_covariant_loop()

	def __build_spectral_loop(self):
		t3_covs = T.dtensor3('covs')
		
		# Get eigenvectors
		egmatrix_t3, update0 = theano.scan(
			fn=lambda covariance: T.nlinalg.eig(convariance)[0],
			sequences=[t3_covs]
		)
		
		# Transpose is equal to the inverse
		egmatrix_inv_t3, update1 = theano.scan(
			fn=lambda egv: egv.dimsuffle(1,0),
			sequences=[egmatrix]
		)

		# Get identity matrix
		identity = T.identity_like(egvector_inv_loop)

		# Get eigenvalues
		egvalues_t2, update2 = theano.scan(
			fn=lambda covariance: T.nlinalg.eig(convariance)[1],
			sequences=[t3_covs]
		)
		
		egvalue_v = T.dvector('egv')

		diagnol_root_t2, update3 = theano.scan(
			fn=lambda row, egvalues: row * egvalues,
			sequences=[identity],
			non_sequences=[egvalue_v]
		)

		diagnol_root_t2, update4 = theano.scan(
			fn=lambda matrix,
		)


	def __build_covariant_loop(self):
		df = T.dvector('f')
		covariant_loop, updates = theano.scan(
			fn=lambda x, row: row * x,
			sequences=[df],
			non_sequences=[df]
		)

		return function(inputs=[df],outputs=[covariant_loop])
		
		#self.patch = self.__build_patches()

	# Let resolution be a number 0 < resolution <= 1
	# Where we do a bilateral patching. Resolution of
	# 1 means that 
	# def __build_patches(self):
	# 	t_patches = T.dtensor4('patches')
	# 	t_imgs = T.dtensor3('imgs')
	# 	t_imgv = T.dmatrix('imgv')
	# 	t_res = T.iscalar('res')
	# 	t_psize = T.iscalar('psize')
	# 	t_x = T.iscalar('x')

	# 	def build_patch(p_indx, p_imgv, t_x, p_res, p_psize):
	# 		y = p_imgv[t_x: p_psize]
	# 		t_x += p_res
	# 		return y

	# 	def bl(p_imgv):
	# 		# This should return all the patches
	# 		return ae(p_imgv,p_x,p_res,p_size)

	# 	steps = int((self.size / 2 - self.patch_size / 2) / self.resolution)

	# 	rp, _ = theano.scan(
	# 		fn = build_patch,
	# 		sequences=[T.arange(steps)],
	# 		outputs_info=None,
	# 		non_sequences=[t_imgv, t_x, t_res, t_psize]
	# 	)

	# 	ae = function([t_imgv, t_x, t_res, t_psize],on_unused_input='warn',outputs=[rp])

	# 	def sdf(p_imgv, p_x, p_res, p_psize):
	# 		return function(inputs=[p_imgv, p_x, p_res, p_psize],on_unused_input='warn',outputs=[rp])

	# 	v_x = T.iscalar('v_x')
	# 	v_m = T.iscalar('v_r')
	# 	v_u = T.iscalar('v_p')

	# 	# This should return all the patches
	# 	row_patches, _ = theano.scan(
	# 		fn=sdf,
	# 		outputs_info=None,
	# 		sequences=[t_imgs],
	# 		non_sequences=[v_x, v_m, v_u]
	# 	)

		# patches, updates = theano.scan(
		# 	fn = build_patch,
		# 	outputs_info= None,
		# 	sequences=[row_patches, T.arange(steps)],
		# 	non_sequences=[t_res, t_psize]
		# )
		
		#return function(inputs=[z, asdf],outputs=[testr])
		#return function(inputs=[t_imgs, t_x, t_res, t_psize],outputs=[row_patches])

		# k = T.iscalar("k")
		# A = T.vector("A")

		# # Symbolic description of the result
		# result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
		#                               outputs_info=T.ones_like(A),
		#                               non_sequences=A,
		#                               n_steps=k)

		# # We only care about A**k, but scan has provided us with A**1 through A**k.
		# # Discard the values that we don't care about. Scan is smart enough to
		# # notice this and not waste memory saving them.
		# final_result = result[-1]

		# # compiled function that returns A**k
		# power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

		# print power(range(10),2)
		# print power(range(10),4)

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
			)
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
					for img_idx in range(number_of_imgs):
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
		for img_idx in range(number_of_imgs):
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


	def covariance(self, sps, u):
		cov_t3 = np.zeros(
			(
				sps.shape[0],
				sps.shape[1] * sps.shape[2],
				sps.shape[1] * sps.shape[2]
			)
		)

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
		dp = dp.reshape((sps.shape[0], self.patch_size ** 2)).eval()

		for i in range(sps.shape[0]):
			cov_t3[i] = self.covariant_loop(dp[i])[0]

		return T.mean(cov_t3, axis=0)

	def get_covariance_subs(self, patches, mean):
		number_of_imgs = patches.shape[0] / (self.stride * self.stride * 4)
		sub_patches = []
		u = None
		covariance_subs = np.zeros(
			(
				4 * (self.stride ** 2),
				self.patch_size ** 2,
				self.patch_size ** 2
			)
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
					for img_idx in range(number_of_imgs):
						sub_patches[img_idx] = patches[self.map_patch_idx(img_idx, 0, i, j)]
						u = mean[self.map_patch_idx(0,sb,i,j)]
					# Now we can calculate our covariance
					covariance_subs[self.map_patch_idx(0,sb,i,j)] = self.covariance(sub_patches, u).eval()
		return covariance_subs
	# This applies a gaussian blur on our image
	def blur(self, imgv):
		return ndimage.gaussian_filter(imgv, self.sigma)

	# Whitens and max-suppresses our image
	def whiten(self, imgv):
		return