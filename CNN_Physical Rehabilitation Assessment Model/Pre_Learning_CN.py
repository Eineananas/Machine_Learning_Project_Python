import os
from openpyxl import load_workbook
import re
import pandas as pd
import numpy as np
import torch


# 文件夹路径
output_path = 'C:/Users/Desktop/对齐数据/tensor'
file_path = 'C:/Users/Desktop/对齐数据'
# 获取文件夹下所有文件
file_names = os.listdir(file_path)

# 读取数据
os.chdir(file_path)
# 排序文件名（按编号顺序）
file_names.sort(key=lambda x: int(x[1:]))
# 遍历并打开Excel文件
for file_name in file_names:
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, file_name)
    # 检查文件是否是Excel文件
    if file_name.endswith('.xlsx'):
        # 打开Excel文件
        workbook = load_workbook(file_path)
        sheet = workbook.active
        num= re.findall(r'\d+', file_name)
        df = pd.read_excel(sheet,header=None)
        df1=df.iloc[:,0:45]
        df2=df.iloc[:,54:99]
        #左脚XYZ
        dataframe1 = df1.iloc[:, ::3]  # 1, 4, 7... 列
        dataframe2 = df1.iloc[:, 1::3]  # 2, 5, 8... 列
        dataframe3 = df1.iloc[:, 2::3]  # 3, 6, 9... 列
        #右脚XYZ
        dataframe4 = df2.iloc[:, ::3]  
        dataframe5 = df2.iloc[:, 1::3]  
        dataframe6 = df2.iloc[:, 2::3]  
        dataframes = [dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, dataframe6]
        # 将每个DataFrame转换为PyTorch张量，并放入一个列表中
        tensor_list = [torch.Tensor(df.values) for df in dataframes]
        
        # 使用torch.stack将张量列表堆叠成一个新的张量，axis=2表示沿着第三个维度堆叠
        result_tensor = torch.stack(tensor_list, axis=2)
        result_tensor=result_tensor[1000:14000,:,:]
        # 输出PyTorch张量的形状
        print("PyTorch张量的形状：", result_tensor.shape)
        #print(result_tensor[:, :, :6])
        # 切分Tensor并保存为文件
        sliced_tensors = torch.chunk(result_tensor, 13, dim=0)
        # 保存切分后的Tensor为文件
        for i, sliced_tensor in enumerate(sliced_tensors):
            torch.save(sliced_tensor, os.path.join(output_path, f'{i + num*100}.pt'))
      
        # 关闭Excel文件
        workbook.close()



import torch
from torch.utils.data import Dataset, DataLoader
import os

path = 'C:/Users/WeiTh/Desktop/对齐数据/tensor'

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label = int(file_name.split('.')[0])
        label=label//100
        label = torch.tensor(label).float()
        # 文件名即为标签
        file_path = os.path.join(self.data_folder, file_name)
        tensor = torch.load(file_path)
        tensor=torch.transpose(tensor, 0, 2)  # 将第1维和第3维交换，变为 [batch_size, 3, 15, 1000]
        return tensor, label

# 创建自定义数据集实例
dataset = CustomDataset(path)


from torch.utils.data import random_split
# 假设 dataset 是你的 CustomDataset 对象，包含了所有的数据
dataset_size = len(dataset)
#train_size = int(0.8 * dataset_size)  # 使用80%的数据作为训练集
train_size = 96  # 使用80%的数据作为训练集
test_size = dataset_size - train_size  # 剩下的作为测试集
# 随机分割数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# 现在 train_dataset 包含了80%的数据，test_dataset 包含了20%的数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=21, shuffle=False)


for inputs, labels in train_loader:
    #inputs = torch.transpose(inputs, 1, 3)  # 将第1维和第3维交换，变为 [batch_size, 3, 15, 1000]
    print(inputs.size())
    print(labels.size())

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        #输出：16*15*1000
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        #输出：16*8*501
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 输出：32*8*501
        # 定义全连接层
        self.fc1 = nn.Linear(32 * 5 * 251, 128)  # 输入通道数是32*8*8，因为经过两次池化后，图片大小变成了1/4
        self.fc2 = nn.Linear(128, 1)  # 输出为一个连续变量

    def forward(self, x):
        # 前向传播函数
        #relu层的设置
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        #print("Ilo,",x.size())
        x = x.view(-1, 32 * 5 * 251)  # 将特征图展平成一维向量
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型实例
model = CNNModel()

# 打印模型结构
print(model)

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 创建损失函数
criterion = nn.MSELoss()

print(enumerate(train_loader))

num_epochs = 20
# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 将输入数据转换为torch的Variable
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels)
        labels = labels.unsqueeze(1)
        #size从torch.Size([32])变成torch.Size([32，1])
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印训练信息
        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

#在测试集上验证
model.eval()
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for inputs, labels in test_loader:
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels)
        labels = labels.unsqueeze(1)
        outputs = model(inputs)
        # 计算预测值
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += labels.size(0)
    average_loss = total_loss / len(test_loader)
    #输出测试集上的损失
    print('Average Test Loss: {:.4f}'.format(average_loss))




