import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
labels = pd.read_csv('labels.csv')
breed = set(labels['breed'])
n = len(labels)
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))
img_size = 256
x = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# inputs

for i in range(n):
    x[i] = cv2.resize(cv2.imread('./train/%s.jpg' % labels['id'][i]), (img_size, img_size))
    y[i][class_to_num[labels['breed'][i]]] = 1

#test_labels = pd.read_csv('sample_submission.csv')
#n_test = len(test_labels)
#x_test = np.zeros((n_test, img_size, img_size, 3), dtype=np.uint8)

# for i in range(n_test):
 #   x_test[i] = cv2.resize(cv2.imread('./test/%s.jpg' % test_labels['id'][i], (img_size, img_size)))

# Normalizing

x = np.array(x, np.float32) / 255
print("initialised succesfully")
# x_test = np.array(x_test, np.float32) / 255

# train_test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
k = 24
l = 36
m = 48
p = 64
q = 64


def new_weights(a, b, c, d):
    w = tf.Variable(tf.truncated_normal([a, b, c, d], stddev=0.1))
    b_ = tf.Variable(tf.ones([d])/10)
    return w, b_

w1, b1 = new_weights(5, 5, 3, k)
w2, b2 = new_weights(5, 5, k, l)
w3, b3 = new_weights(5, 5, l, m)
w4, b4 = new_weights(3, 3, m, p)
w5, b5 = new_weights(3, 3, p, q)

n = 11640
w6 = tf.Variable(tf.truncated_normal([16*16*q, n], stddev=0.1))
b6 = tf.Variable(tf.ones([11640])/10)
w7 = tf.Variable(tf.truncated_normal([n, 1000], stddev=0.1))
b7 = tf.Variable(tf.ones([1000])/10)
w8 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.1))
b8 = tf.Variable(tf.ones([500])/10)
w9 = tf.Variable(tf.truncated_normal([500, 120], stddev=0.1))
b9 = tf.Variable(tf.ones([120])/10)
x_ = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='x_')
y_ = tf.placeholder(tf.float32, shape=[None, 120])
lr = tf.placeholder(tf.float32)
y1 = tf.nn.relu(tf.nn.conv2d(x_, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)
y4 = tf.nn.relu(tf.nn.conv2d(y3, w4, strides=[1, 2, 2, 1], padding='SAME') + b4)
y5 = tf.nn.relu(tf.nn.conv2d(y4, w5, strides=[1, 2, 2, 1], padding='SAME') + b5)
yy = tf.reshape(y5, shape=[-1, 16*16*q])
y6 = tf.nn.relu(tf.matmul(yy, w6) + b6)
y7 = tf.nn.relu(tf.matmul(y6, w7) + b7)
y8 = tf.nn.relu(tf.matmul(y7, w8) + b8)
y = tf.nn.softmax(tf.matmul(y8, w9) + b9)
sess = tf.Session()
init = tf.global_variables_initializer()
true = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
accuracy = tf.reduce_mean(tf.cast(true, tf.float32))
optmizer = tf.train.GradientDescentOptimizer(lr)
train_step = optmizer.minimize(cross_entropy)
sess.run(init)
max_learning_rate = 0.0003
min_learning_rate = 0.0001
decay_speed = 2000.0
batch_size = 10
for i in range(8000):
		
    	learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    	for bid in range(int(len(x) / batch_size)):
        	x_batch = x[bid * batch_size:(bid + 1) * batch_size]
        	y_batch = y[bid * batch_size:(bid + 1) * batch_size]
        	train_data = {x_: x_batch, y_: y_batch, lr: learning_rate}
        	sess.run(train_step, feed_dict=train_data)
    	if i % 100 == 0:
        	test_data = {x_: X_test, y_: y_test}
        	print(sess.run(accuracy, feed_dict=test_data))
