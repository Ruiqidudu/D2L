import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.sum().backward()
            trainer.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            print(f'epoch {epoch + 1}, train loss {train_loss:.4f}, '
                  f'test loss {test_loss:.4f}')
    print('学到的权重:', net[0].weight.data.numpy())
    print('真实权重前4项:', true_w[:4].numpy())


if __name__ == '__main__':
    print('=' * 50)
    print('1. 欠拟合：只用前2个特征 (x⁰, x¹)')
    print('=' * 50)
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
          labels[:n_train], labels[n_train:])

    print('\n' + '=' * 50)
    print('2. 刚好：用前4个特征 (x⁰~x³)')
    print('=' * 50)
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:])

    print('\n' + '=' * 50)
    print('3. 过拟合：用全部20个特征 (x⁰~x¹⁹)')
    print('=' * 50)
    train(poly_features[:n_train, :], poly_features[n_train:, :],
          labels[:n_train], labels[n_train:])
