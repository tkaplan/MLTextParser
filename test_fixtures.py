import numpy as np
import theano.tensor as T
import theano

t2_123 = np.asarray([
	[1,2,3],
	[4,5,6],
	[7,8,9]
])

t3_123 = np.asarray([
	[
		[1,2,3],
		[4,5,6],
		[7,8,9]
	],
	[
		[10,20,30],
		[40,50,60],
		[70,80,90]
	],
	[
		[100,200,300],
		[400,500,600],
		[700,800,900]
	]
])

t3_0 = np.zeros(
	(
		3,
		3,
		3
	)
)

t3_0 = np.ones(
	(
		3,
		3,
		3
	)
)