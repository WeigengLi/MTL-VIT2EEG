import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointNetLayer, self).__init__()
        
        # MLP for processing each point independently
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch_size, in_channels, num_points)
        return self.mlp(x)

class ModifiedPointNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedPointNet, self).__init__()
        
        self.layer1 = PointNetLayer(2, 64)  # Adjust input channels to 2
        self.layer2 = PointNetLayer(64, 128)
        self.layer3 = PointNetLayer(128, 256)
        
        # Add more layers as needed
        
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch_size, num_points, 2)
        x = x.permute(0, 2, 1)  # Change the order of dimensions for PointNet
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling
        global_features = torch.max(x, dim=2)[0]
        
        # Fully connected layers
        fc1_output = F.relu(self.fc1(global_features))
        output = self.fc2(fc1_output)
        
        return output

# 示例用法
batch_size = 32
num_points = 1024
num_classes = 10

# 创建修改后的PointNet模型
modified_pointnet_model = ModifiedPointNet(num_classes)

# 随机生成输入点序列数据
input_data = torch.rand((batch_size, num_points, 2))

# 将输入数据传递给修改后的PointNet模型
output_data = modified_pointnet_model(input_data)

# 输出数据的形状
print(output_data.shape)
