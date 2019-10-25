import os
import cPickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1).astype(float)/255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self


    def next_batch(self, batch_size):
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]
        self._i = (self._i + batch_size) / len(self.images)
        return x, y


DATA_PATH = "cifar-10-batches-py"
def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as f:
        dict = cPickle.load(f)
    return dict


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1,6)]).load()
        self.test = CifarLoader(["test_batch"]).load()

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) for i in range(size)])
    plt.imshow(im)
    plt.show(block=True)

d = CifarDataManager()
print "Number of train images: {}".format(len(d.train.images))
print "Number of train labels: {}".format(len(d.train.labels))
print "Number of test images: {}".format(len(d.test.images))
print "Number of test images: {}".format(len(d.test.labels))
images = d.train.images
display_cifar(images, 10)

cifar = CifarDataManager()

NUM_STEPS = 1000
MINIBATCH_SIZE = 50

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

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)

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
        acc, _ = sess.run([accuracy, optimizer], feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.9})
        if i % 100 == 0:
            print("Training accuracy at {} step: {:.4}%".format(i, acc*100))

    test(sess)