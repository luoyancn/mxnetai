# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon

def softmax(x): # softmax的作用就是把一个输出转换为概率
    exp = nd.exp(x) # exp函数将x当中的所有数据变更为正数
    partition = exp.sum(axis=1, keepdims=True) # 对第一维的数据求和，即第一列数据
    return exp/partition


def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def SGD(params, lr):
    for param in params:
        param[:] = param - lr*param.grad


def mnistfashion_data(batch_size):

    def _transform(data, label):
        return data.astype('float32')/255, label.astype('float32')


    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=_transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=_transform)

    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    return train_data, test_data


def square_loss(yhat, y):
    return (yhat -  y.reshape(yhat.shape)) ** 2


def load_data_fashion_mnist(batch_size):
    def _transform_mnist(data, label):
        return nd.transpose(
            data.astype('float32'), (2,0,1))/255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=_transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=_transform_mnist)

    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    return (train_data, test_data)


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    for data, label in data_iterator:
        output = net(data.as_in_context(ctx))
        acc += accuracy(output, label.as_in_context(ctx))
    return acc / len(data_iterator)


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx