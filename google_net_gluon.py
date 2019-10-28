# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# Google Net其中就使用了Network in Network的思想，并对其作了很大的改进
# Google Net当中通常有多个4个并行卷积层迭加的块，这个块就称之为Inception。
# 其思路基本如下：

# 1. 输入经过一个1×1的卷积，直接到输出
# 2. 输入经过一个1×1的卷积，再进行3 × 3的卷积，最后到输出
# 3. 输入经过一个1×1的卷积，随后进行5 × 5的卷积，最后到输出
# 4. 输入经过一个3×3的最大池化，再进行1×1的卷积，最后到输出
# 5. 上述四路运算同时进行，所有的输出到最后，进行concat操作，合并并汇集到一起


from mxnet import init
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn

import utils


class Inception(nn.Block):

    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, *args, **kwargs):
        super(Inception, self).__init__(*args, **kwargs)
        with self.name_scope():
            # 第一条路径
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')

            # second path
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')

            # third path
            self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
            self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')

            # fourth path
            self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
            self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)


class GoogleNet(nn.Block):

    def __init__(self, num_classes, verbose=False, *args, **kwargs):
        super(GoogleNet, self).__init__(*args, **kwargs)
        self.verbose = verbose
        with self.name_scope():
            b1 = nn.Sequential()
            b1.add(nn.Conv2D(64, kernel_size=7, strides=2,
                             padding=3, activation='relu'),
                   nn.MaxPool2D(pool_size=3, strides=2))

            b2 = nn.Sequential()
            b2.add(nn.Conv2D(64, kernel_size=1),
                   nn.Conv2D(192, kernel_size=3, padding=1),
                   nn.MaxPool2D(pool_size=3, strides=2))

            b3 = nn.Sequential()
            b3.add(Inception(64, 96, 128, 16, 32, 32),
                   Inception(128, 128, 192, 32, 96, 64),
                   nn.MaxPool2D(pool_size=3, strides=2))

            b4 = nn.Sequential()
            b4.add(Inception(192, 96, 208, 16, 48, 64),
                   Inception(160, 112, 224, 24, 64, 64),
                   Inception(128, 128, 256, 24, 64, 64),
                   Inception(112, 144, 288, 32, 64, 64),
                   Inception(256, 160, 320, 32, 128, 128),
                   nn.MaxPool2D(pool_size=3, strides=2))

            b5 = nn.Sequential()
            b5.add(Inception(256, 160, 320, 32, 128, 128),
                   Inception(384, 192, 384, 48, 128, 128),
                   nn.AvgPool2D(pool_size=2))

            b6 = nn.Sequential()
            b6.add(nn.Flatten(), nn.Dense(num_classes))

            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


train_data, test_data = utils.load_data_fashion_mnist_new(batch_size=64, resize=96)
ctx = utils.try_gpu()
net_ = GoogleNet(10)
net_.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net_.collect_params(), 'sgd', {'learning_rate': 0.01})
utils.train(train_data, test_data, net_, loss, trainer, ctx, num_epochs=1)