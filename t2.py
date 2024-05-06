import torch
import torch.nn as nn

# 定义输入张量
input_data = torch.tensor([[1, 1, 1, 1, 0],
                           [1, 1, 0, 0, 0],
                           [1, 0, 0, 1, 1],
                           [1, 0, 0, 1, 0],
                           [0, 1, 1, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 定义卷积核
conv_kernel = torch.tensor([[[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]]], dtype=torch.float32).unsqueeze(0)

# 定义卷积层
conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1,bias=False)

# 将卷积核加载到卷积层中
conv_layer.weight.data = conv_kernel

# 进行卷积运算
output = conv_layer(input_data)

# 打印卷积输出
print(output.squeeze().detach().numpy())