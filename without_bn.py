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
img_size = 64
x_ = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# inputs

for i in range(n):
    x_[i] = cv2.resize(cv2.imread('./train/%s.jpg' % labels['id'][i]), (img_size, img_size))
    y[i][class_to_num[labels['breed'][i]]] = 1

#test_labels = pd.read_csv('sample_submission.csv')
#n_test = len(test_labels)
#x_test = np.zeros((n_test, img_size, img_size, 3), dtype=np.uint8)

# for i in range(n_test):
 #   x_test[i] = cv2.resize(cv2.imread('./test/%s.jpg' % test_labels['id'][i], (img_size, img_size)))

# Normalizing

x_ = np.array(x_, np.float32) / 255
# x_test = np.array(x_test, np.float32) / 255

# train_test split

X_train, X_test, y_train, y_test = train_test_split(x_, y, test_size=0.1)
k = 24
l = 48
m = 96
def new_weights(a, b, c, d):
    w = tf.Variable(tf.truncated_normal([a, b, c, d], stddev=0.1))
    b_ = tf.Variable(tf.ones([d])/10)
    return w, b_

w1, b1 = new_weights(5, 5, 3, k)
w2, b2 = new_weights(4, 4, k, l)
w3, b3 = new_weights(4, 4, l, m)

n = 800
w4 = tf.Variable(tf.truncated_normal([16*16*m, n], stddev=0.1))
b4 = tf.Variable(tf.ones([400])/10)
w5 = tf.Variable(tf.truncated_normal([n, 120], stddev=0.1))
b5 = tf.Variable(tf.ones([120])/10)

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 120])
lr = tf.placeholder(tf.float32)
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)
yy = tf.reshape(y3, shape=[-1, 16*16*m])
yp = tf.nn.relu(tf.matmul(yy, w4) + b4)
y4 = tf.nn.dropout(yp, 0.75)
y = tf.nn.softmax(tf.matmul(y4, w5) + b5)
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
batch_size = 1000
test_data = {x: X_test, y_: y_test}
for i in range(8000):

    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    for bid in range(int(len(x_) / batch_size)):
        x_batch = X_train[bid * batch_size:(bid + 1) * batch_size]
        y_batch = y_train[bid * batch_size:(bid + 1) * batch_size]
        train_data = {x: x_batch, y_: y_batch, lr: learning_rate}
        sess.run(train_step, feed_dict=train_data)
    print(sess.run(accuracy, feed_dict=test_data))
