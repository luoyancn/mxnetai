import random

from mxnet import ndarray as nd
from mxnet import autograd as ag


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size = 10

x = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_b
y += .01 *nd.random_normal(shape=y.shape)


def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size, num_examples)])
        yield nd.take(x, j), nd.take(y, j)


w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1, ))
params = [w, b]
for param in params:
    param.attach_grad()


def net(x):
    return nd.dot(x, w) + b


def squar_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2


def sgd(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


epochs = 5
learning_rate = .001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with ag.record():
            output = net(data)
            loss = squar_loss(output, label)
        loss.backward()
        sgd(params, learning_rate)
        total_loss += nd.sum(loss).asscalar()
    print('Epoch %d, average loss: %f' %(e, total_loss/num_examples))
