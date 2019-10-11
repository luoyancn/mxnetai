# -*- coding:utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
batch_size = 10

x = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_b
y += .01 *nd.random_normal(shape=y.shape)

dataset = gluon.data.ArrayDataset(x, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
net = gluon.nn.Sequential() # 定义一个空的神经网络（模型），MXNet使用Sequential将所有的层串联起来
net.add(gluon.nn.Dense(1)) # 加入Dense层，Dense必须要定义的参数是输出节点的个数，线性模型当中该参数为1
net.initialize() # 初始化模型权重，使用默认的随机初始化方法
squar_loss = gluon.loss.L2Loss() # 方差损值函数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1}) # 优化，使用Trainer实现参数的梯度下降

# 进行训练
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with ag.record():
            output = net(data)
            loss = squar_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print('Epoch %d, average loss: %f' %(e, total_loss/num_examples))

dense_layer = net[0] # 获取网络当中的第一层，实际上可以通过索引获取网络的每一层
weight = dense_layer.weight.data() # 获取对应网络层的权重参数
bias = dense_layer.bias.data() # 获取对应网络层的bias参数

print(true_w, weight)
print(true_b, bias)
