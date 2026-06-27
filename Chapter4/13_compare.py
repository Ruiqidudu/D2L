import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def train_and_record(net, train_iter, test_iter, loss, trainer, num_epochs):
    """训练模型并记录每个 epoch 的 train_loss, train_acc, test_acc"""
    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            metric.add(l.sum(), d2l.accuracy(y_hat, y), y.numel())
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        net.eval()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        test_accs.append(test_acc)

        print(f'epoch {epoch + 1}, loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}, test acc {test_acc:.4f}')
    return train_losses, train_accs, test_accs


def make_net(dropout1=None, dropout2=None):
    """创建三层网络，可选择是否加 Dropout"""
    layers = [nn.Flatten(), nn.Linear(784, 256), nn.ReLU()]
    if dropout1 is not None:
        layers.append(nn.Dropout(dropout1))
    layers.append(nn.Linear(256, 256))
    layers.append(nn.ReLU())
    if dropout2 is not None:
        layers.append(nn.Dropout(dropout2))
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)


def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_epochs = 40
    lr = 0.1
    loss = nn.CrossEntropyLoss(reduction='none')

    # ========== 1. 普通三层（无正则化）==========
    print('=' * 50)
    print('1. 普通三层（无正则化）')
    print('=' * 50)
    net1 = make_net()
    net1.apply(init_weights)
    trainer1 = torch.optim.SGD(net1.parameters(), lr=lr)
    _, train_accs1, test_accs1 = train_and_record(
        net1, train_iter, test_iter, loss, trainer1, num_epochs)

    # ========== 2. 普通三层 + L2 正则化 ==========
    print('=' * 50)
    print('2. 普通三层 + L2 正则化（wd=0.001）')
    print('=' * 50)
    net2 = make_net()
    net2.apply(init_weights)
    trainer2 = torch.optim.SGD(net2.parameters(), lr=lr, weight_decay=0.001)
    _, train_accs2, test_accs2 = train_and_record(
        net2, train_iter, test_iter, loss, trainer2, num_epochs)

    # ========== 3. 普通三层 + Dropout ==========
    print('=' * 50)
    print('3. 普通三层 + Dropout（0.2, 0.5）')
    print('=' * 50)
    net3 = make_net(dropout1=0.2, dropout2=0.5)
    net3.apply(init_weights)
    trainer3 = torch.optim.SGD(net3.parameters(), lr=lr)
    _, train_accs3, test_accs3 = train_and_record(
        net3, train_iter, test_iter, loss, trainer3, num_epochs)

    # ========== 画图对比 ==========
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(12, 5))

    # 左图：Test Accuracy 对比
    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_accs1, 'r-o', label='plain (no reg)')
    plt.plot(epochs, test_accs2, 'b-s', label='plain + L2 (wd=0.001)')
    plt.plot(epochs, test_accs3, 'g-^', label='plain + Dropout')
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.title('Test Accuracy Comparison')

    # 右图：Train Accuracy 对比
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs1, 'r-o', label='plain (no reg)')
    plt.plot(epochs, train_accs2, 'b-s', label='plain + L2 (wd=0.001)')
    plt.plot(epochs, train_accs3, 'g-^', label='plain + Dropout')
    plt.xlabel('epoch')
    plt.ylabel('train accuracy')
    plt.legend()
    plt.title('Train Accuracy Comparison')

    plt.tight_layout()
    plt.savefig('C:/Users/Dudu/PythonProject/deeplearning_learn/Chapter4/三种方法对比.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
