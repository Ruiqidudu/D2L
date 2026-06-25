import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))


# nn是神经网络的缩写


net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01) #均值0，标准差0.01
net[0].bias.data.fill_(0)# b 为0

loss = nn.MSELoss()#均方误差

trainer = torch.optim.SGD(net.parameters(), lr=0.03) #参数为w1,w2,b 学习率0.03

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y) #net(x)为线性函数 批次误差
        trainer.zero_grad()
        l.backward() #前馈
        trainer.step()
    l = loss(net(features), labels) #真实误差（全体误差）
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)