# -*- coding:utf-8 -*-

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


def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output,label)
    return acc/len(data_iterator)


def mnistfashion_data(batch_size):

    def _transform(data, label):
        return data.astype('float32')/255, label.astype('float32')


    mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=_transform)
    mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=_transform)

    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    
    return train_data, test_data