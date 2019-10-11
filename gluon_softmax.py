# -*- coding:utf-8 -*-

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd as ag
from matplotlib import pyplot as plt

import utils


batch_size = 256
train_data, test_data = utils.mnistfashion_data(batch_size)
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))

net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with ag.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print('Epoch %d, loss: %f, Train acc %f, test acc %f' % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
