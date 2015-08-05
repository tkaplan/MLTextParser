import Utils.DataLoader.NOData.NOFont as NOFont
import Utils.DataLoader.NOData.NOHandwritting as NOHandwritting

import scipy.ndimage as ndimage
import scipy.misc as misc

__dim__ = 28

a = NOFont.NOFont(.5,.4)
b = NOHandwritting.NOHandwritting(.5,.4)

assert a.get_classpath('a') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample037'
assert a.get_classpath('9') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample010'
assert a.get_classpath('8') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample009'
assert a.get_classpath('0') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample001'
assert a.get_classpath('A') == '/Users/tkaplan/MLTextParser/TrainingData/Font/Sample011'

imgv = a.get_characterset('A').next_test()
resize = __dim__ / float(imgv.shape[0]) if imgv.shape[0] > imgv.shape[1] else __dim__ / float(imgv.shape[1]) 
img_resample = misc.imresize(imgv, resize, 'bilinear', 'P')
misc.imsave("test1.png",img_resample)

#print b.get_characterset('B').next_test().shape