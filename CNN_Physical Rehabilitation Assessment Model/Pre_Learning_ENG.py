import os
from openpyxl import load_workbook
import re
import pandas as pd
import numpy as np
import torch

# Folder paths
output_path = 'C:/Users/Desktop/AlignData/tensor'
file_path = 'C:/Users/Desktop/AlignData'
# Get all files in the folder
file_names = os.listdir(file_path)

# Read data
os.chdir(file_path)
# Sort file names (in numerical order)
file_names.sort(key=lambda x: int(x[1:]))
# Iterate through Excel files
for file_name in file_names:
    # Construct full file path
    file_path = os.path.join(file_path, file_name)
    # Check if the file is an Excel file
    if file_name.endswith('.xlsx'):
        # Open Excel file
        workbook = load_workbook(file_path)
        sheet = workbook.active
        num = re.findall(r'\d+', file_name)
        df = pd.read_excel(sheet, header=None)
        df1 = df.iloc[:, 0:45]
        df2 = df.iloc[:, 54:99]
        # Extract left foot XYZ
        dataframe1 = df1.iloc[:, ::3]
        dataframe2 = df1.iloc[:, 1::3]
        dataframe3 = df1.iloc[:, 2::3]
        # Extract right foot XYZ
        dataframe4 = df2.iloc[:, ::3]
        dataframe5 = df2.iloc[:, 1::3]
        dataframe6 = df2.iloc[:, 2::3]
        dataframes = [dataframe1, dataframe2, dataframe3, dataframe4, dataframe5, dataframe6]
        # Convert each DataFrame to PyTorch tensors and store in a list
        tensor_list = [torch.Tensor(df.values) for df in dataframes]

        # Stack tensors along the third axis using torch.stack (axis=2)
        result_tensor = torch.stack(tensor_list, dim=2)
        result_tensor = result_tensor[1000:14000, :, :]
        # Output the shape of the PyTorch tensor
        print("PyTorch Tensor Shape:", result_tensor.shape)
        # Slice the tensor and save as files
        sliced_tensors = torch.chunk(result_tensor, 13, dim=0)
        # Save sliced tensors as files
        for i, sliced_tensor in enumerate(sliced_tensors):
            torch.save(sliced_tensor, os.path.join(output_path, f'{i + num * 100}.pt'))

        # Close the Excel file
        workbook.close()



import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label = int(file_name.split('.')[0])
        label = label // 100
        label = torch.tensor(label).float()
        file_path = os.path.join(self.data_folder, file_name)
        tensor = torch.load(file_path)
        tensor = torch.transpose(tensor, 0, 2)  # Swap dimensions to [batch_size, 3, 15, 1000]
        return tensor, label

dataset = CustomDataset(output_path)
dataset_size = len(dataset)
train_size = 96
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=21, shuffle=False)


#CNN model configuration
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 251, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 251)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))


#Test-Set
model.eval()
with torch.no_grad():
    total_loss = 0
    total_samples = 0
    for inputs, labels in test_loader:
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels).unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += labels.size(0)
    average_loss = total_loss / len(test_loader)
    print('Average Test Loss: {:.4f}'.format(average_loss))
