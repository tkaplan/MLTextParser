import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import Utils.ImgPreprocessing.ImgPreprocessoring
from Utils.ImgPreprocessing.ImgPreprocessing import PreProcessing as ImgPP

import scipy.ndimage as ndimage
import scipy.misc as misc

from theano import tensor as T
from theano import function

import numpy as np

a = NOFont.NOFont(.5,.4)
b = NOHandwritting.NOHandwritting(.5,.4)

assert a.get_classpath('a') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample037'
assert a.get_classpath('9') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample010'
assert a.get_classpath('8') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample009'
assert a.get_classpath('0') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample001'
assert a.get_classpath('A') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample011'

csetA = a.get_characterset('A')

# We instantiate our image preprocessor
pp = ImgPP(size=32, patch_size=8, sigma=1.3, resolution=3)

# t3 data blurred
t3 = csetA.t3_training(pp)

# imgv = csetA.next_training()
# imgv = pp.scale(imgv)
# imgv = pp.blur(imgv)
# print imgv.shape
# print pp.get_patch(imgv)
#print t3.shape