import pickle 
import numpy as np
from numpy import array
import tensorflow as tf



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


with open("dog_breed.pkl", 'rb') as file1:
	tmp = pickle.load(file1)
	train_data ,test_data = tmp[0], tmp[1]

train = np.array(train_data['scaled'].tolist())
test = np.array(test_data['scaled'].tolist())
file1.close()
with open('label.pkl','rb') as file2:
	tmp1 = pickle.load(file2)
labels = np.array(tmp.tolist())
del(tmp)
del(tmp1)
file2.close()




