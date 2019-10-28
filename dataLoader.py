import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print(tf.__version__)


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
        x, y = self.images[int(self._i):int(self._i)+batch_size], \
               self.labels[int(self._i):int(self._i)+batch_size]
        self._i = (self._i + batch_size) % len(self.images)
        return x, y


DATA_PATH = "cifar-10-batches-py"


def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
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
    # plt.show(block=True)


# d = CifarDataManager()
# print("Number of train images: {}".format(len(d.train.images)))
# print("Number of train labels: {}".format(len(d.train.labels)))
# print("Number of test images: {}".format(len(d.test.images)))
# print("Number of test images: {}".format(len(d.test.labels)))
# images = d.test.images
# display_cifar(images, 10)