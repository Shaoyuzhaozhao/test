"""
ResNet-TS Model for WiFi Sensing
时间序列专用的ResNet模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSBlock(nn.Module):
    """时间序列专用的ResNet块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(TSBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetTS(nn.Module):
    """
    ResNet-TS: 时间序列专用的ResNet
    适配频谱图输入，将2D频谱图转换为1D时间序列处理
    """
    def __init__(self, input_shape=(1, 129, 47), num_classes=9):
        super(ResNetTS, self).__init__()
        
        # 输入维度处理：将频谱图转换为时间序列
        # 频谱图 (1, 129, 47) -> 时间序列 (129, 47)
        self.input_dim = input_shape[1]  # 129 (频率维度)
        self.seq_len = input_shape[2]    # 47 (时间维度)
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-TS层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(TSBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(TSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 支持频谱图输入格式转换
        if x.dim() == 4 and x.shape[1:] == (1, 129, 47):
            x = x.squeeze(1)  # (batch, 1, 129, 47) -> (batch, 129, 47)
        elif x.dim() == 3 and x.shape[1:] == (129, 47):
            pass  # 已经是正确格式
        else:
            # 其他格式的处理
            if x.dim() == 3:
                x = x.permute(0, 2, 1)  # (batch, seq_len, features) -> (batch, features, seq_len)
        
        # 确保输入格式为 (batch, features, seq_len)
        # 对于频谱图，features=129(频率), seq_len=47(时间)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x