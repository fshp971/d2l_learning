# Chapter 3 Linear Neural Networks

## Linear Regression
A naive linear regression implementation and test with random data.



## The Fashion-MNIST Dataset

[Official Site](https://github.com/zalandoresearch/fashion-mnist)

One should downloads the 4 dataset files manually, they are the main datasets that used in this chapter.

According to the file formats described on the [original MNIST dataset website](http://yann.lecun.com/exdb/mnist/), a reading-data-module has been implemented and store in `Dataset.py`. One can use the module to read and iterate samples.



## Softmax Regression

### Softmax Function

![cross_entropy_loss](../Figures/softmax_function.svg)

Here $x, w_i, b_i$ may be vectors, and $w_i, b_i$ is the parameters going to be optimized.

### Cross Entropy Loss

![cross_entropy_loss](../Figures/cross_entropy_loss.svg)

The formula above shows that its computation can be slightly optimized.



## Multilayer Perceptron (MLP)

### Activation function

![cross_entropy_loss](../Figures/activation.svg)

### MLP with multiple hidden layers

![cross_entropy_loss](../Figures/mlp.svg)

In MLP, the initialization of parameters (i.e. W, b) is essential. One should manually initialize them before optimization with random values, for example, values of normal distribution.