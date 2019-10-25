# -*- coding:utf-8 -*-
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import image
from mxnet import nd
from mxnet.gluon import nn

import utils


net = nn.Sequential()

with net.name_scope():
    net.add(nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))

    net.add(nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))

    net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))

    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation='relu'))
    net.add(nn.Dropout(.5))

    net.add(nn.Dense(4096, activation='relu'))
    net.add(nn.Dropout(.5))

    # 如果是使用imagenet数据集，则dense参数应该为1000
    # 此处使用的数据集为比较小的数据集
    net.add(nn.Dense(10))


def transform(data, label):
    data = image.imresize(data, 224, 224)
    return utils.transform_mnist(data, label)


batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(
    batch_size, transform=transform)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})

for epoch in range(1):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net, ctx)
    print('Epoch %d, Loss %f, Train acc %f, Test acc %f ' % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))