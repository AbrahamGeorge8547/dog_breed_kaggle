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
img_size = 32
x = np.zeros((n, img_size, img_size, 3), dtype=np.uint8)
y = np.zeros((n, n_class), dtype=np.uint8)

# inputs

for i in range(n):
    x[i] = cv2.resize(cv2.imread('./train/%s.jpg' % labels['id'][i]), (img_size, img_size))
    y[i][class_to_num[labels['breed'][i]]] = 1
x = np.array(x, np.float32) / 255
# x_test = np.array(x_test, np.float32) / 255

# train_test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

tf.set_random_seed(0.0)

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 120])
lr = tf.placeholder(tf.float32)
tst = tf.placeholder(tf.bool)
itera = tf.placeholder(tf.int32)


def batchnorm(ylogits, is_test, iteration, offset, convolution=False):
    exp_moving_average = tf.train.ExponentialMovingAverage(0.999, iteration)
    epsilon = 1e-5
    if convolution:
        mean, variance = tf.nn.moments(ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(ylogits, [0])
    update_moving_averages = exp_moving_average.apply([mean, variance])
    m1 = tf.cond(is_test, lambda: exp_moving_average.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_average.average(variance), lambda: variance)
    ybn = tf.nn.batch_normalization(ylogits, m1, v, offset, None, epsilon)
    return ybn, update_moving_averages


k = 16
l = 32
m = 64
n = 200
w1 = tf.Variable(tf.truncated_normal([6, 6, 3, k], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, tf.float32, [k]))
w2 = tf.Variable(tf.truncated_normal([5, 5, k, l], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, tf.float32, [l]))
w3 = tf.Variable(tf.truncated_normal([4, 4, l, m], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, tf.float32, [m]))

w4 = tf.Variable(tf.truncated_normal([8 * 8 * m, n], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, tf.float32, [n]))
w5 = tf.Variable(tf.truncated_normal([n, 120], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, tf.float32, [120]))

y1l = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
y1bn, update_ema1 = batchnorm(y1l, tst, itera, b1, convolution=True)
y1 = tf.nn.relu(y1bn)

y2l = tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME')
y2bn, update_ema2 = batchnorm(y2l, tst, itera, b2, convolution=True)
y2 = tf.nn.relu(y2bn)

y3l = tf.nn.conv2d(y2, w3, strides=[1, 2, 2, 1], padding='SAME')
y3bn, update_ema3 = batchnorm(y3l, tst, itera, b3, convolution=True)
y3 = tf.nn.relu(y3bn)


yy = tf.reshape(y3, shape=[-1, 8 * 8 * m])

y4l = tf.matmul(yy, w4)
y4bn, update_ema4 = batchnorm(y4l, tst, itera, b4)
y4 = tf.nn.relu(y4bn)

y1logits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(y1logits)
update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
print(np.size(y1logits))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y1logits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 1000
x_train = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
y_train = np.zeros((batch_size, n_class), dtype=np.uint8)
for i in range(10000):

    tf.reset_default_graph()
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(- i / decay_speed)
    for bid in range(int(len(x) / batch_size)):
        x_batch = X_train[bid * batch_size:(bid + 1) * batch_size]
        y_batch = y_train[bid * batch_size:(bid + 1) * batch_size]
        print("a batch was taken in")
        sess.run(train_step, feed_dict={X: x_batch, y_: y_batch, lr: learning_rate, tst: False})
        sess.run(update_ema, feed_dict={X: x_batch, y_: y_batch, lr: learning_rate, tst: False, itera: i})
    print(sess.run(accuracy, feed_dict={X: X_test, y_: y_test, tst: True}))
