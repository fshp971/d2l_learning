import os
import torch as tf
import numpy as np

class Reader:
    # One should put the correct MNIST data set files
    # under the location corresponding to 'path'.

    def __init__(self, path = './'):
        # Read train data from files.
        path1 = path + 'train-images-idx3-ubyte'
        path2 = path + 'train-labels-idx1-ubyte'
        self.train_features, self.train_labels = self.ReadDataSet(path1, path2)

        # Read test data from files.
        path1 = path + 't10k-images-idx3-ubyte'
        path2 = path + 't10k-labels-idx1-ubyte'
        self.test_features, self.test_labels = self.ReadDataSet(path1, path2)

    def Byte2Int(src):
        res = 0
        for x in src:
            res = (res<<8) + x
        return res

    # path1 stand for features data file, and path2 stand for labels data file.
    def ReadDataSet(self, path1, path2):
        fp1 = open(path1, 'rb')
        fp2 = open(path2, 'rb')

        # Read magic number.
        fp1.read(4)
        fp2.read(4)

        n = Reader.Byte2Int( fp1.read(4) )
        assert n == Reader.Byte2Int( fp2.read(4) )
        row, col = Reader.Byte2Int(fp1.read(4) ), Reader.Byte2Int( fp1.read(4) )

        features = tf.empty(n, row*col, dtype = tf.float32)
        labels = tf.empty(n, 1, dtype = tf.float32)

        for i in range(n):
            #print('i = %d\n', i)
            features[i] = tf.tensor([fp1.read(row*col)], dtype = tf.float32)
            labels[i] = tf.tensor([fp2.read(1)], dtype = tf.float32)

        return features, labels

    def IterTrain(self, batch_size = 50):
        idx = [i for i in range(len(self.train_labels))]
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = tf.LongTensor( idx[i : min(i+batch_size, len(idx))] )
            yield self.train_features.index_select(0, j), self.train_labels.index_select(0, j)

    def IterTest(self, batch_size = 50):
        idx = [i for i in range(len(self.test_labels))]
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            j = tf.LongTensor( idx[i : min(i+batch_size, len(idx))] )
            yield self.test_features.index_select(0, j), self.test_labels.index_select(0, j)
