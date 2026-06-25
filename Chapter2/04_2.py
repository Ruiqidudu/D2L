import os
import pandas as pd
import torch

# 确保 data/ 目录存在
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

# 写入 CSV 文件
file_path = os.path.join(data_dir, 'house_tiny.csv')
with open(file_path, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')        # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

print(f'CSV 文件已保存到 {file_path}')

data = pd.read_csv(file_path)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 只对数值列 NumRooms 填充均值，避免对字符串列 Alley 计算均值
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print(X)
print(y)
