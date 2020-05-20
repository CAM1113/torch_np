import torch

x = torch.tensor([i for i in range(4 * 3 * 3)]).view(4,3,3)
print(x)

# 内存连续性不遍
z = x.view(3,3,4)
print(z)

# 维度变换，改变了内存的连续性
y = x.permute(1,2,0)
print(y)