from NOData import NOBase
from os import path

class NOHandwritting(NOBase):
	def __init__(self, training, validation):
		NOBase.__init__(self, training, validation)

	def get_classpath(self, character):
		return path.join(
			path.abspath('.'),
			'TrainingData',
			'Images',
			'GoodImg',
			'Bmp',
			'Sample0{0}'
		).format(self.get_classno(character))