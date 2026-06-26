import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train_concise(wd, train_losses, test_losses):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},#加上L2范数
        {"params": net[0].bias}], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 10 == 0:
            train_loss = evaluate_loss(net, train_iter, loss)
            test_loss = evaluate_loss(net, test_iter, loss)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f'epoch {epoch + 1}, train loss {train_loss:.4f}, '
                  f'test loss {test_loss:.4f}')
    print(f'w的L2范数（wd={wd}）：{net[0].weight.norm().item():.4f}')
    print()


if __name__ == '__main__':
    # 训练并记录 loss
    train_losses_0, test_losses_0 = [], []
    train_losses_3, test_losses_3 = [], []

    print('=' * 50)
    print('1. 没有 L2 正则化（wd=0）')
    print('=' * 50)
    train_concise(0, train_losses_0, test_losses_0)

    print('=' * 50)
    print('2. 有 L2 正则化（wd=3）')
    print('=' * 50)
    train_concise(3, train_losses_3, test_losses_3)

    # 画图
    epochs = list(range(10, 101, 10))
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_0, 'r-o', label='train (wd=0)')
    plt.plot(epochs, test_losses_0, 'r--s', label='test (wd=0)')
    plt.plot(epochs, train_losses_3, 'b-o', label='train (wd=3)')
    plt.plot(epochs, test_losses_3, 'b--s', label='test (wd=3)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Train/Test Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_losses_0, 'r--s', label='test (wd=0)')
    plt.plot(epochs, test_losses_3, 'b--s', label='test (wd=3)')
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.legend()
    plt.title('Test Loss Only')

    plt.tight_layout()
    plt.savefig('C:/Users/Dudu/PythonProject/deeplearning_learn/Chapter4/l2_对比图.png', dpi=150)
    plt.show()
