#from __future__ import print_function
import numpy as np
import torch
import time

batch_size = 50
lr = 0.01
epochs = 600

def LineReg(X, w, b):
    return torch.mm(X,w) + b

def Loss(y1, y2):
    return ( (y1-y2) ** 2 ).mean()

def Optimizer(params, lr, bsz): # Stochastic Gradient Descent
    for param in params:
        param.data -= lr * param.grad / bsz

'''
def test():
    a = torch.ones(3,5)
    print(id(a))
    print(a)
    for i in range(3):
        yield a.index_select(0, torch.LongTensor([i]))
        print(a)
'''

def Read():
    m = int(input())
    n = int(input())
    X = torch.Tensor(m, n)
    Y = torch.Tensor(m, 1)
    w0 = torch.Tensor(n, 1)
    #b0 = 0
    b0 = torch.Tensor(1, 1)
    for i in range(m):
        for k in range(n):
            X[i][k] = float( input() )
        Y[i][0] = float( input() )
    for i in range(n):
        w0[i][0] = float( input() )
    #b0 = float( input() )
    b0[0][0] = float( input() )
    #print(len(X))
    #print(len(X[0]))
    return m, n, X, Y, w0, b0

def IterData(batch_size, X, Y):
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    for i in range(0, len(X), batch_size):
        j = torch.LongTensor(index[i: min(i+batch_size, len(X))])
        yield X.index_select(0, j), Y.index_select(0, j)

'''
w = torch.tensor(np.random.normal(0, 0.01, (2, 1)), dtype=torch.float)
print(w)
'''

m, n, features, labels, w0, b0 = Read()

w = torch.zeros(n, 1, requires_grad = True)
b = torch.zeros(1, 1, requires_grad = True)
#w.requires_grad_(requires_grad = True)
#b.requires_grad_(requires_grad = True)

err0 = (((torch.mm(features, w0) + b0) - labels) ** 2).mean()

for epoch in range(epochs):
    for X, Y in IterData(batch_size, features, labels):
        loss = Loss( LineReg(X, w, b), Y )
        loss.backward()
        Optimizer([w,b], lr, len(X))
        '''
        print('w =', w)
        #print(w.grad)
        print('')
        print('b =', b)
        print('\n')
        '''
        w.grad.data.zero_()
        b.grad.data.zero_()
    hhh = Loss( LineReg(features, w, b), labels)
    print('Epoch = %d, Loss = %f, Loss2 = %f' % (epoch, hhh, hhh-err0) )

err = (((torch.mm(features, w) + b) - labels) ** 2).mean()

print([w0, b0])
print('err0 =', err0, '\n')

print([w, b])
print('err =', err, '\n')
