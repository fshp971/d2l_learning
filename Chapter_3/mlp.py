import torch as tf
import numpy as np

import Dataset

def Dropout(X, drop_prob):
    if drop_prob == 1:
        return tf.zeros_like(X) * X
    return (tf.FloatTensor(X.size()).uniform_(0,1) > drop_prob).float() * X / (1-drop_prob)

def Softmax(X):
    X_exp = X.exp()
    div = X_exp.sum(dim = 1, keepdim = True)
    return X_exp / div

def CrossEntropyLoss(X, Y):
    return - tf.log( X.gather(1, Y.view(len(X),1)) ).sum() #/ len(X)

def mlpNet(W1, B1, W2, B2, drop_prob, X):
    # return tf.mm( (tf.mm(X, W1) + B1).tanh(), W2 ) + B2
    # return tf.mm( Dropout( (tf.mm(X, W1) + B1).tanh(), drop_prob ), W2 ) + B2
    return tf.mm( Dropout( (tf.mm(X, W1) + B1).relu(), drop_prob ), W2 ) + B2

def CalcStat(W1, B1, W2, B2, drop_prob, X, Y, label_number):
    result = Softmax( mlpNet(W1, B1, W2, B2, drop_prob, X) ).argmax(dim=1)
    stat = [0 for i in range(label_number)]
    cnt = [0 for i in range(label_number)]
    for i in range(len(X)):
        cnt[Y[i].item()] += 1
        if Y[i].item() == result[i].item(): stat[Y[i].item()] += 1
    acc = [stat[i]/cnt[i] for i in range(label_number)]
    return acc

def CalcAccuracy(W1, B1, W2, B2, drop_prob, X, Y):
    return (Softmax( mlpNet(W1, B1, W2, B2, drop_prob, X) ).argmax(dim=1) == Y).sum().item() / len(X)

def main():
    epochs = 10
    lr = 0.000002
    batch_size = 256
    drop_prob = 0.5

    reader = Dataset.Reader()
    dim_inputs, dim_outputs, dim_hidden = reader.row*reader.col, reader.label_number, 256

    # random initialization is essential
    W1 = tf.tensor(np.random.normal(0, 0.01, (dim_inputs, dim_hidden)), dtype = tf.float32, requires_grad = True)
    B1 = tf.tensor(np.random.normal(0, 0.01, (1, dim_hidden)), dtype = tf.float32, requires_grad = True)
    W2 = tf.tensor(np.random.normal(0, 0.01, (dim_hidden, dim_outputs)), dtype = tf.float32, requires_grad = True)
    B2 = tf.tensor(np.random.normal(0, 0.01, (1, dim_outputs)), dtype = tf.float32, requires_grad = True)

    train_loss = CrossEntropyLoss(Softmax(mlpNet(W1, B1, W2, B2, drop_prob, reader.train_features)), reader.train_labels)
    train_acc = CalcAccuracy(W1, B1, W2, B2, drop_prob, reader.train_features, reader.train_labels )
    test_acc = CalcAccuracy(W1, B1, W2, B2, drop_prob, reader.test_features, reader.test_labels )
    print('epoch = %d, train_loss = %.4f, train_acc = %.4f, test.acc = %.4f' % 
            (0, train_loss, train_acc, test_acc))

    for epoch in range(epochs):
        for X, Y in reader.IterTrain(batch_size):
            loss = CrossEntropyLoss( Softmax( mlpNet(W1, B1, W2, B2, drop_prob, X) ), Y )
            loss.backward()

            for param in [W1,B1,W2,B2]:
                param.data -= lr * param.grad
                param.grad.data.zero_()

        train_loss = CrossEntropyLoss(Softmax(mlpNet(W1, B1, W2, B2, drop_prob, reader.train_features)), reader.train_labels)
        train_acc = CalcAccuracy(W1, B1, W2, B2, drop_prob, reader.train_features, reader.train_labels )
        test_acc = CalcAccuracy(W1, B1, W2, B2, drop_prob, reader.test_features, reader.test_labels )
        print('epoch = %d, train_loss = %.4f, train_acc = %.4f, test.acc = %.4f' % 
                (epoch+1, train_loss, train_acc, test_acc))

    res1 = CalcStat(W1, B1, W2, B2, drop_prob, reader.train_features, reader.train_labels, reader.label_number)
    res2 = CalcStat(W1, B1, W2, B2, drop_prob, reader.test_features, reader.test_labels, reader.label_number)
    print()
    print('lr = %f, batch_size = %d, drop_prob = %f' % (lr, batch_size, drop_prob))
    print()
    print('rate of pictures in train set that are correctly predicted as its ground truth label')
    print(res1)

    print()
    print('rate of pictures in test set that are correctly predicted as its ground truth label')
    print(res2)


if __name__ == '__main__':
    main()
