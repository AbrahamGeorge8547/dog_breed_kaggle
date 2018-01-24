import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import math
labels = pd.read_csv('labels.csv')
breed = set(labels['breed'])
n = len(labels)
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))
img_size = 224
x = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# inputs

for i in range(n):
    x[i] = cv2.resize(cv2.imread('./train/%s.jpg' % labels['id'][i]), (img_size, img_size))
    y[i][class_to_num[labels['breed'][i]]] = 1
x = np.array(x, np.float32) / 255

print("Train test split complete")
k = 16
l = 32
m = 64


def new_weights(a, b, c, d):
    w = tf.Variable(tf.truncated_normal([a, b, c, d], stddev=0.1))
    b_ = tf.Variable(tf.ones([d])/10)
    return w, b_


w1, b1 = new_weights(2, 2, 3, k)
w2, b2 = new_weights(2, 2, k, l)
w3, b3 = new_weights(2, 2, l, m)
w4 = tf.Variable(tf.truncated_normal([56*56*m, 1024], stddev=0.1))
b4 = tf.Variable(tf.ones([1024])/10)
w5 = tf.Variable(tf.truncated_normal([1024, 120], stddev=0.1))
b5 = tf.Variable(tf.ones([120])/10)
x_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None, 120])
lr = tf.placeholder(tf.float32)

y1 = tf.nn.relu(tf.nn.conv2d(x_, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME') + b3)
yy = tf.reshape(y3, shape=[-1, 56*56*m])
y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
y5 = tf.nn.softmax(tf.matmul(y4, w5) + b5)
sess = tf.Session()
init = tf.global_variables_initializer()

true = tf.equal(tf.arg_max(y_, 1), tf.argmax(y5, 1))
test2 = tf.argmax(y5,1)
cross_entropy = -tf.reduce_sum(y_*tf.log(y5))
accuracy = tf.reduce_mean(tf.cast(true, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)
sess.run(init)
print(sess.run(y5))
max_learning_rate = 0.0003
min_learning_rate = 0.0001
decay_speed = 2000.0
batch_size = 10
for i in range(8000):
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
    for bid in range(int(len(x) / batch_size)):
        batch_x = x[bid * batch_size:(bid + 1) * batch_size]
        batch_y = y[bid * batch_size:(bid + 1) * batch_size]
    train_data = {x_: batch_x, y_: batch_y, lr: learning_rate}
    test_data = {x_:batch_x, y_:batch_y}
    sess.run(train_step, feed_dict=train_data)
    print(sess.run(accuracy,feed_dict = test_data))


    



























