import pickle
import numpy as np
import tensorflow as tf
import os
import random


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x1, w):
    return tf.nn.conv2d(x1, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x1):
    return tf.nn.max_pool(x1, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input2, shape):
    w = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input2, w) + b)


def full_layer(input1, size):
    in_size = int(input1.get_shape()[1])
    w = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input1, w) + b


with open("dog_breed.pkl", 'rb') as file1:
        tmp = pickle.load(file1)
        train_data, test_data = tmp[0], tmp[1]

train = np.array(train_data['scaled'].tolist())
test = np.array(test_data['scaled'].tolist())
file1.close()
with open('label.pkl', 'rb') as file2:
        tmp1 = pickle.load(file2)
labels = np.array(tmp1.tolist())
del tmp
del tmp1
file2.close()
batch_size = 360
step_size = 3000
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 120])
keep_prob = tf.placeholder(tf.float32)
if os.path.exists("model.pkl"):
    with open('model.pkl', 'rb') as mod_file:
        tmp = pickle.load(mod_file)
        sess, y_conv = tmp[0], tmp[1]
    del tmp
    mod_file.close()
else:
    conv1 = conv_layer(x, [5, 5, 3, 32])
    conv1_pool = max_pool_2x2(conv1)
    conv2 = conv_layer(conv1_pool, [5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)
    conv2_flat = tf.reshape(conv2_pool,[-1, 8*8*64])
    fully_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
    full1_drop = tf.nn.dropout(fully_1, keep_prob=keep_prob)
    y_conv = full_layer(full1_drop, 120)
    classes = tf.argmax(y_conv, 1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(classes, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(step_size):
        index = random.sample(range(len(train)), batch_size)
        x_batch, y_batch = np.array(train[[index]].tolist()), np.array(labels)[[index]]
        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.2})
        if i % 100 == 0:
            acc = np.mean([sess.run(accuracy, feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.2})])
            print("Steps:" + str(i) + ", acc:" + str(acc))
    with open(r"model_dogbreed.pkl", "wb") as output_file:
        pickle.dump([sess, y_conv], output_file)












































