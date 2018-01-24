import pickle
import tensorflow as tf
import numpy as np
import random
import math


k = 16
l = 32
n = 2048
tf.set_random_seed(0.0)


def new_weights(a,b,c,d):
    w = tf.Variable(tf.truncated_normal([a,b,c,d], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, tf.float32, [d]))
    return w, b



w1, b1 = new_weights(3, 3, 3, k)
w2, b2 = new_weights(4, 4, k, l)

w4 = tf.Variable(tf.truncated_normal([8*8*l, n], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, tf.float32, [n]))
w5 = tf.Variable(tf.truncated_normal([n, 120], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, tf.float32, [120]))


with open("dog_breed32.pkl", 'rb') as file1:
    tmp = pickle.load(file1)
    train_data, test_data = tmp[0], tmp[1]

train = np.array(train_data['scaled'].tolist())
test = np.array(test_data['scaled'].tolist())
file1.close()
with open('label32.pkl', 'rb') as file2:
    tmp1 = pickle.load(file2)
labels = np.array(tmp1.tolist())
del tmp
del tmp1
file2.close()


x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 120])
lr = tf.placeholder(tf.float32)
tst = tf.placeholder(tf.bool)
iteration = tf.placeholder(tf.int32)
pkeep_conv = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1 ,0 ,0 ,1]) + tf.constant([0 ,1 ,1 ,0])
    return noiseshape


def batch_norm(ylogits, is_test, itera, offset, convolution=False):
    exp_m_avg = tf.train.ExponentialMovingAverage(0.99, itera)
    epsilon = 1e-5
    if convolution:
        mean, variance = tf.nn.moments(ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(ylogits, [0])
    update_mvg = exp_m_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_m_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_m_avg.average(variance), lambda: variance)
    y_batch_norm = tf.nn.batch_normalization(ylogits, m, v, offset, None, epsilon)
    return y_batch_norm, update_mvg


y1l = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
y1m = tf.nn.max_pool(y1l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
y1bn, umavg1 = batch_norm(y1m, tst, iteration, b1, convolution=True)
y1r = tf.nn.relu(y1bn)
y1 = tf.nn.dropout(y1r, pkeep_conv, compatible_convolutional_noise_shape(y1r))


y2l = tf.nn.conv2d(y1, w2, strides=[1, 1, 1, 1], padding='SAME')
y2m = tf.nn.max_pool(y2l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
y2bn, umavg2 = batch_norm(y2m, tst, iteration, b2, convolution=True)
y2r = tf.nn.relu(y2bn)
y2 = tf.nn.dropout(y2r, pkeep_conv, compatible_convolutional_noise_shape(y2r))


yy = tf.reshape(y2, [-1, 8*8*l])
y4l = tf.matmul(yy, w4) + b4
y4bn, umavg4 = batch_norm(y4l, tst, iteration, b4)
y4r = tf.nn.relu(y4bn)
y4 = tf.nn.dropout(y4r, pkeep)


y5 = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(y5)
update_ema = tf.group(umavg1, umavg2, umavg4)

saver = tf.train.Saver()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y5, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.cast(correct_pred, tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = 500
for i in range(250):
    index = random.sample(range(len(train)), batch_size)
    x_batch, y_batch = np.array(train[[index]].tolist()), np.array(labels)[[index]]
    max_lr = 0.02
    min_lr = 0.0001
    decay_speed = 500
    learning_rate = min_lr + (max_lr-min_lr)*math.exp(-i/decay_speed)
    sess.run(train_step, {x: x_batch, y_: y_batch, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1 })
    sess.run(update_ema, {x: x_batch, y_: y_batch, tst: False, iteration: i, pkeep_conv: 1, pkeep: 1})
    if i %10 == 0:
	    acc = np.mean([sess.run(accuracy, feed_dict={x: train, y_: labels, tst: True, pkeep_conv:1, pkeep: 1})])
	    print("Steps:" + str(i) + ", acc:" + str(acc))

saver.save(sess, 'model.meta')


























































