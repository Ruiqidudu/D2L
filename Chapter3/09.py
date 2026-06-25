import torch
from torch import nn
from d2l import torch as d2l


def evaluate_accuracy(net, data_iter, device):
    """计算模型在数据集上的准确率"""
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(axis=1) == y).sum().item()
            total += y.numel()
    return correct / total


if __name__ == '__main__':
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net = net.to(device)  # 将模型搬到GPU

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights);

    loss = nn.CrossEntropyLoss(reduction='mean')

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 15

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        count = 0
        num_batches = 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 将数据搬到GPU
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            train_loss += l.item()
            train_acc += (net(X).argmax(axis=1) == y).sum().item()
            count += y.numel()
            num_batches += 1
        test_acc = evaluate_accuracy(net, test_iter, device)
        print(f'epoch {epoch + 1}, loss {train_loss/num_batches:.4f}, train acc {train_acc/count:.4f}, test acc {test_acc:.4f}')
