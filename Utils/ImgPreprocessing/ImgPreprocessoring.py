import scipy.ndimage as ndimage
import scipy.misc as misc

class PreProcessing:
	def __init__(self, size, blur):
		self.size = size
		self.blur = blur

	def normalize(self, imgv):
		# We want to shorten by the largest dimmension
		resize = size / float(imgv.shape[0]) if imgv.shape[0] > imgv.shape[1] else size / float(imgv.shape[1]) 
		# We then resize the image
		imresize = misc.imresize(imgv, resize, 'bilinear', 'P')
		# The next step is to super impose this onto some background
		# Or image, however we don't care to do that yet
		return imresize

	# This applies a gaussian blur on our image
	def blur(self, imgv):
		return

	# Whitens and max-suppresses our image
	def whiten(self, imgv):
		return