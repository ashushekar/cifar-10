import dataLoader
import tensorflow as tf
import numpy as np

cifar = dataLoader.CifarDataManager()

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape=shape)
    b = bias_variable(shape=[shape[3]])
    return tf.nn.relu(tf.nn.bias_add(conv2d(input, W), b))

def full_layer(input, size):
    W = weight_variable(shape=[int(input.get_shape()[1]), size])
    b = bias_variable(shape=[size])
    return tf.nn.bias_add(tf.matmul(input, W), b)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1 = maxpool2d(conv1)

conv2 = conv_layer(conv1, shape=[5, 5, 32, 64])
conv2 = maxpool2d(conv2)

conv2_flat = tf.reshape(conv2, shape=[-1, 8*8*64])
flat_layer = tf.nn.relu(full_layer(conv2_flat, size=1024))

flat_layer_dropout = tf.nn.dropout(flat_layer, keep_prob=keep_prob)

y_pred = full_layer(flat_layer_dropout, size=10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_true: Y[i], keep_prob: 1.0}) for i in range(10)])
    print("Accuracy: {:.4}%".format(acc * 100))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(NUM_STEPS):
        batch = cifar.train.next_batch(MINIBATCH_SIZE)
        acc, _ = sess.run([accuracy, optimizer], feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            print("Training accuracy at {} step: {:.4}%".format(i, acc*100))

    test(sess)