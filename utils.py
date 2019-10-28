# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import image
from mxnet import autograd

import numpy as np

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


def load_data_fashion_mnist(batch_size, datasets='datasets', transform=None):

    def _transform_mnist(data, label):
        return nd.transpose(
            data.astype('float32'), (2,0,1))/255, label.astype('float32')

    if None == transform:
        transform = _transform_mnist

    mnist_train = gluon.data.vision.FashionMNIST(
        root=datasets, train=True, transform=transform)
    mnist_test = gluon.data.vision.FashionMNIST(
        root=datasets, train=False, transform=transform)

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


def transform_mnist(data, label):
    # change data from height x weight x channel to channel x height x weight
    return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')


class DataLoader(object):
    """similiar to gluon.data.DataLoader, but faster"""
    def __init__(self, X, y, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X = X
        self.y = y

    def __iter__(self):
        n = self.X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.X = nd.array(self.X.asnumpy()[idx])
            self.y = nd.array(self.y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            yield (self.X[i*self.batch_size:(i+1)*self.batch_size],
                   self.y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return self.X.shape[0]//self.batch_size

def load_data_fashion_mnist_new(batch_size, resize=0, datasets='datasets'):
    """download the fashion mnist dataest and then load into memory"""
    def _transform_mnist(data, label):
        if resize:
            # resize to resize x resize
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

    mnist_train = gluon.data.vision.FashionMNIST(
        root=datasets, train=True, transform=_transform_mnist)[:]
    mnist_test = gluon.data.vision.FashionMNIST(
        root=datasets, train=False, transform=_transform_mnist)[:]

    train_data = DataLoader(mnist_train[0], nd.array(mnist_train[1]), batch_size, shuffle=True)
    test_data = DataLoader(mnist_test[0], nd.array(mnist_test[1]), batch_size, shuffle=False)
    return (train_data, test_data)


def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        for i, batch in enumerate(train_data):
            data, label = _get_batch(batch, ctx)
            with autograd.record():
                output = net(data)
                L = loss(output, label)
            L.backward()

            trainer.step(data.shape[0])

            train_loss += nd.mean(L).asscalar()
            train_acc += accuracy(output, label)

            n = i + 1
            if print_batches and n % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    n, train_loss/n, train_acc/n))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/n, train_acc/n, test_acc))


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)