"""
RFNet Model for WiFi Sensing
专为WiFi感知设计的网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFNet(nn.Module):
    """
    RF-Net: 专为WiFi感知设计的网络架构
    适用于频谱图输入 (batch, channels, height, width)
    """
    def __init__(self, input_shape=(1, 129, 47), num_classes=9):
        super(RFNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 计算全连接层输入维度
        self._calculate_fc_input_dim(input_shape)
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_fc_input_dim(self, input_shape):
        """计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self.fc_input_dim = x.view(1, -1).size(1)
    
    def forward(self, x):
        # 支持频谱图输入格式转换
        if x.dim() == 4 and x.shape[1:] == (1, 129, 47):
            x = x.squeeze(1).unsqueeze(1)  # 确保正确的通道维度
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
            
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x