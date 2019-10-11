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


net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten()) # 拉直，即将多维数组转换为(1 ,n)的形状
    net.add(gluon.nn.Dense(num_hidden, activation='relu'))
    net.add(gluon.nn.Dense(num_outputs))

net.initialize()
train_data, test_data = utils.mnistfashion_data(batch_size)
softmax_corss_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_corss_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print('Epoch %d, Loss: %f, Train acc %f, Test acc %f' %(
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
