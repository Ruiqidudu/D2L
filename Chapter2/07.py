import torch

x = torch.arange(4.0)

print(x)
print(x.shape)

x.requires_grad_(True)

y = 2 * torch.dot(x, x)

y.backward()
print("梯度:")
print(x.grad)

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = 2*(x * x)

print(y.shape)
print(y)

# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)