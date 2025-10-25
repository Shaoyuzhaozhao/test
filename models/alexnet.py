"""
AlexNet Model for WiFi Sensing
适配频谱图输入的AlexNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet for WiFi Sensing
    适配频谱图输入 (batch, channels, height, width)
    """
    def __init__(self, input_shape=(1, 129, 47), num_classes=9):
        super(AlexNet, self).__init__()
        
        # 适配频谱图输入的特征提取层
        self.features = nn.Sequential(
            # 第一层卷积 - 适配频谱图尺寸
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二层卷积
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第三层卷积
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第五层卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # 自适应池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 动态计算全连接层输入维度
        self._calculate_fc_input_dim(input_shape)
        
        # 分类器 - 减少参数量以适配频谱图任务
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.fc_input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def _calculate_fc_input_dim(self, input_shape):
        """动态计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.features(x)
            x = self.avgpool(x)
            self.fc_input_dim = x.view(1, -1).size(1)
    
    def forward(self, x):
        # 支持频谱图输入格式转换
        if x.dim() == 4 and x.shape[1:] == (1, 129, 47):
            x = x.squeeze(1).unsqueeze(1)  # 确保正确的通道维度
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x