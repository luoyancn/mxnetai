# -*- coding:utf-8 -*-

from mxnet import autograd as ag
from mxnet import gluon
from mxnet import ndarray as nd

import utils


num_inputs = 28 * 28
num_outputs = 10
num_hidden = 256
weight_scale = .01
learning_rate = .5
batch_size = 256

w1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale) # scale表示标准差
b1 = nd.zeros(num_hidden)
w2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)
params = [w1, b1, w2, b2]
for param in params:
    param.attach_grad()


def relu(x):
    return nd.maximum(x, 0) # 自定义ReLU的实现


# 定义模型，将计算层与激活函数串联起来
def net(x):
    x = x.reshape((-1, num_inputs))
    h1 = relu(nd.dot(x, w1) + b1) # 只对第一层进行激活操作
    output = nd.dot(h1, w2) + b2  # 输出层不做激活操作
    return output


# 由于自定义的softmax和交叉熵数值不稳定，因此直接使用内置的
softmax_corss_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
train_data, test_data = utils.mnistfashion_data(batch_size)
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_corss_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print('Epoch %d, Loss: %f, Train acc %f, Test acc %f' %(
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
