from NeuralNetwork import NeuralNetwork as NN
import numpy as np
import test_fixtures as tf

a = tf.t3_123

assert NN.pad_with_zeros(a, 10).shape[0] % 10 == 0
assert NN.pad_with_zeros(a, 3).shape[0] % 3 == 0
assert NN.pad_with_zeros(a, 5).shape[0] % 5 == 0
assert NN.pad_with_zeros(a, 1).shape[0] % 1 == 0
assert NN.pad_with_zeros(a, 2).shape[0] % 2 == 0

assert NN.pad_with_wrap(a, 10).shape[0] % 10 == 0
assert NN.pad_with_wrap(a, 3).shape[0] % 3 == 0
assert NN.pad_with_wrap(a, 5).shape[0] % 5 == 0
assert NN.pad_with_wrap(a, 1).shape[0] % 1 == 0
assert NN.pad_with_wrap(a, 2).shape[0] % 2 == 0

assert NN.DataSet.training.value == 0
assert NN.DataSet.validation.value == 1
assert NN.DataSet.testing.value == 2