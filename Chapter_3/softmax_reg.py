import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch as tf
import os
import errno

import Dataset

def Softmax(X):
    X_exp = X.exp()
    div = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / div

# Y can be seen as index
def CrossEntropyLoss(X,Y):
    assert len(X) == len(Y)
    return - tf.log( X.gather(1, Y.view(len(X),1)) ).sum() / len(X)

def CalcAccuracy(W, B, X, Y):
    ok_num = (Softmax(tf.mm(X,W) + B).argmax(dim=1) == Y).sum().item()
    return ok_num / len(X)

# Print every picture in X into folder that corresponding to its predicted labels.
# For example:
#   if a pic's id number is ID, predicted label is P, ground truth label is T,
#   then it will be print to the following location:
#       ./class_{P}/pic_{ID}_P{P}_T{T}.png
def Write2Png(W, B, X, Y, label_number):
    try:
        for i in range(label_number):
            os.mkdir('class_' + str(i))
    except OSError as exc:
        if exc.errno != errno.EEXIST: raise
        pass

    result = (Softmax(tf.mm(X,W) + B).argmax(dim=1))
    counter_lim = 10000
    for i in range(len(X)):
        if i >= counter_lim:
            break
        pred = str(result[i].item())
        gtrue = str(Y[i].item())

        cnt[Y[i].item()] += 1
        if pred == gtrue: stat[Y[i].item()] += 1

        mpimg.imsave('./class_' + pred + '/' + 'pic_' + str(i) +
                     '_P' + pred + '_T' + gtrue + '.png', 
                     X[i].view(28,28), cmap=plt.cm.gray)

# acc[i] stand for the rate of pictures with label i
# that be correctly predicted as i
def CalcStat(W, B, X, Y, label_number):
    result = (Softmax(tf.mm(X,W) + B).argmax(dim=1))
    stat = [0 for i in range(label_number)]
    cnt = [0 for i in range(label_number)]
    for i in range(len(X)):
        cnt[Y[i].item()] += 1
        if Y[i].item() == result[i].item(): stat[Y[i].item()] += 1
    acc = [stat[i]/cnt[i] for i in range(label_number)]
    return acc

def main():
    # parameters that can be adjusted
    epochs = 10
    lr = 0.000001
    batch_size = 50

    reader = Dataset.Reader()

    # then the size of W is [10, 28*28]
    W = tf.zeros(reader.row*reader.col, reader.label_number, dtype = tf.float32, requires_grad = True)
    B = tf.zeros(1, reader.label_number, dtype = tf.float32, requires_grad = True)

    # calculate and print the loss and accuracy of initial parameters
    train_loss = CrossEntropyLoss( Softmax(tf.mm(reader.train_features, W) + B), reader.train_labels )
    train_acc = CalcAccuracy( W, B, reader.train_features, reader.train_labels )
    test_acc = CalcAccuracy( W, B, reader.test_features, reader.test_labels )
    print('epoch = %d, train_loss = %.4f, train_acc = %.4f, test.acc = %.4f' % 
            (0, train_loss, train_acc, test_acc))

    for epoch in range(epochs):
        for X, Y in reader.IterTrain(batch_size):
            loss = CrossEntropyLoss( Softmax(tf.mm(X, W) + B), Y )
            loss.backward()

            # Stochastic Gradient Descent
            W.data -= lr * W.grad
            B.data -= lr * B.grad

            W.grad.data.zero_()
            B.grad.data.zero_()

        train_loss = CrossEntropyLoss( Softmax(tf.mm(reader.train_features, W) + B), reader.train_labels )
        train_acc = CalcAccuracy( W, B, reader.train_features, reader.train_labels )
        test_acc = CalcAccuracy( W, B, reader.test_features, reader.test_labels )
        print('epoch = %d, train_loss = %.4f, train_acc = %.4f, test.acc = %.4f' % 
                (epoch+1, train_loss, train_acc, test_acc))

    # Write2Png(W, B, reader.test_features, reader.test_labels, reader.label_number)
    res1 = CalcStat(W, B, reader.train_features, reader.train_labels, reader.label_number)
    res2 = CalcStat(W, B, reader.test_features, reader.test_labels, reader.label_number)
    print()
    print('rate of pictures in train set that are correctly predicted as its ground truth label')
    print(res1)

    print()
    print('rate of pictures in test set that are correctly predicted as its ground truth label')
    print(res2)


if __name__ == '__main__':
    main()
