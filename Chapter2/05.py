import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)
print('x + y =', x + y)
print('x * y =', x * y)
print('x / y =', x / y)
print('x ** y =', x**y)


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:')
print(A)
print('A + B:')
print(A + B)

print('A * B（逐元素相乘）:')
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('a + X（广播机制，标量加三维张量）:')
print(a + X)
print('a * X 的形状:', (a * X).shape)


# ========== 降维 ==========

# 计算元素之和
x = torch.arange(4, dtype=torch.float32)
print('x:', x)
print('x.sum():', x.sum())

print('A.shape:', A.shape)
print('A.sum()（所有元素之和）:', A.sum())

A_sum_axis0 = A.sum(axis=0)
print('A.sum(axis=0)（按列求和）:', A_sum_axis0)
print('形状:', A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print('A.sum(axis=1)（按行求和）:', A_sum_axis1)
print('形状:', A_sum_axis1.shape)

print('A.sum(axis=[0, 1])（所有维度求和）:', A.sum(axis=[0, 1]))

print('A.mean()（均值）:', A.mean())
print('A.sum() / A.numel()（均值）:', A.sum() / A.numel())

# ========== 非降维求和（保持维度） ==========
sum_A = A.sum(axis=1, keepdims=True)
print('A.sum(axis=1, keepdims=True)（按行求和，保持二维）:')
print(sum_A)
print('形状:', sum_A.shape)

print('A / sum_A（广播机制，每行除以该行之和）:')
print(A / sum_A)

# ========== 点积 ==========
y = torch.ones(4, dtype=torch.float32)
print('x:', x)
print('y:', y)
print('torch.dot(x, y)（点积）:', torch.dot(x, y))
print('torch.sum(x * y)（逐元素乘后求和，等价于点积）:', torch.sum(x * y))


# ========== 矩阵向量积 ==========
# A 是 5x4，取第0行作为向量，做矩阵向量积
vec = A[0, :]  # 形状 (4,)
print('A[0, :]（第0行）:', vec)
print('torch.mv(A, vec)（矩阵向量积）:', torch.mv(A, vec))

# ========== 矩阵乘法 ==========
B = torch.ones(4, 3)
print('B（4x3 全1矩阵）:')
print(B)
print('torch.mm(A, B)（矩阵乘法 5x4 @ 4x3 = 5x3）:')
print(torch.mm(A, B))

# ========== 范数 ==========
u = torch.tensor([3.0, -4.0])
print('u:', u)
print('torch.norm(u)（L2范数，√(3²+4²)=5.0）:', torch.norm(u))
print('torch.abs(u).sum()（L1范数，|3|+|-4|=7）:', torch.abs(u).sum())

print('torch.norm(ones(4,9))（矩阵的Frobenius范数，√(36*1)=6）:', torch.norm(torch.ones((4, 9))))
