"""
Enhanced InceptionTime-Attention Model for WiFi Sensing
基于InceptionTime和注意力机制的增强WiFi感知模型
从rfboost_enhanced.py提取的优化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectrogramInceptionModule2D(nn.Module):
    """专门为频谱图设计的Inception模块，增强时频域特征提取"""
    def __init__(self, in_channels, bottleneck_channels=32):
        super().__init__()
        # 1x1卷积降维
        self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        
        # 多尺度时频卷积分支 - 减少分支数量以降低内存占用
        # 时间维度卷积 (频率固定，时间变化) - 减少到2个分支
        self.time_convs = nn.ModuleList([
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(1, k), padding='same')
            for k in [3, 7]  # 减少时间尺度分支
        ])
        
        # 频率维度卷积 (时间固定，频率变化) - 减少到2个分支
        self.freq_convs = nn.ModuleList([
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(k, 1), padding='same')
            for k in [3, 7]  # 减少频率尺度分支
        ])
        
        # 时频联合卷积 - 减少到1个分支
        self.timefreq_convs = nn.ModuleList([
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=(3, 3), padding='same')
        ])
        
        # 池化分支
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        )
        
        # 总输出通道数: 减少分支数量以降低内存占用
        # 时间分支: 2个 (原3个), 频率分支: 2个 (原3个), 时频分支: 1个 (原2个), 池化分支: 1个
        # 总计: 2 + 2 + 1 + 1 = 6个分支
        total_out_channels = bottleneck_channels * 6
        self.bn = nn.BatchNorm2d(total_out_channels)
        self.activation = nn.GELU()  # 使用GELU激活函数，对频谱图更友好
        
        # 动态通道注意力机制
        self.dynamic_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_out_channels, total_out_channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(total_out_channels // 8, total_out_channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(total_out_channels // 4, total_out_channels, 1),
            nn.Sigmoid()
        )

        # 自适应权重生成器
        self.adaptive_weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_out_channels, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),  # 生成3个权重：时间、频率、时频联合
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)

        # 时间维度特征
        time_outs = [conv(x_bottleneck) for conv in self.time_convs]

        # 频率维度特征
        freq_outs = [conv(x_bottleneck) for conv in self.freq_convs]

        # 时频联合特征
        timefreq_outs = [conv(x_bottleneck) for conv in self.timefreq_convs]

        # 池化特征
        pool_out = self.maxpool_conv(x)

        # 拼接所有特征
        out = torch.cat(time_outs + freq_outs + timefreq_outs + [pool_out], dim=1)
        out = self.bn(out)
        out = self.activation(out)

        # 生成自适应权重
        adaptive_weights = self.adaptive_weight_generator(out)  # [B, 3, 1, 1]
        time_weight = adaptive_weights[:, 0:1, :, :]
        freq_weight = adaptive_weights[:, 1:2, :, :]
        timefreq_weight = adaptive_weights[:, 2:3, :, :]

        # 分别处理不同类型的特征
        time_features = torch.cat(time_outs, dim=1)
        freq_features = torch.cat(freq_outs, dim=1)
        timefreq_features = torch.cat(timefreq_outs + [pool_out], dim=1)

        # 动态加权融合
        weighted_features = (time_weight * time_features +
                           freq_weight * freq_features +
                           timefreq_weight * timefreq_features)

        # 应用动态通道注意力
        channel_attention = self.dynamic_channel_attention(out)
        
        # 调整weighted_features的通道数以匹配out
        # 使用1x1卷积将weighted_features从96通道扩展到288通道
        if not hasattr(self, 'channel_adapter'):
            self.channel_adapter = nn.Conv2d(weighted_features.size(1), out.size(1), 1).to(out.device)
        
        weighted_features_adapted = self.channel_adapter(weighted_features)
        final_out = out * channel_attention + weighted_features_adapted * 0.3  # 残差连接

        return final_out





class DynamicSpatialAttention(nn.Module):
    """动态空间注意力模块，根据输入特征动态调整注意力模式"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channels = channels

        # 静态注意力分支（原有功能）
        self.static_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

        # 动态注意力生成器
        self.dynamic_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.GELU(),
            nn.Conv2d(channels//8, 2, 1),  # 生成动态权重
            nn.Sigmoid()
        )

        # 频率-时间分离注意力
        self.freq_attention = nn.Conv2d(channels, 1, kernel_size=(kernel_size, 1),
                                       padding=(kernel_size//2, 0), bias=False)
        self.time_attention = nn.Conv2d(channels, 1, kernel_size=(1, kernel_size),
                                       padding=(0, kernel_size//2), bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 静态注意力（原有逻辑）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        static_attention = torch.cat([avg_out, max_out], dim=1)
        static_attention = self.sigmoid(self.static_conv(static_attention))

        # 动态权重生成
        dynamic_weights = self.dynamic_generator(x)  # [B, 2, 1, 1]
        freq_weight = dynamic_weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        time_weight = dynamic_weights[:, 1:2, :, :]  # [B, 1, 1, 1]

        # 频率和时间维度的动态注意力
        freq_attention = self.sigmoid(self.freq_attention(x))
        time_attention = self.sigmoid(self.time_attention(x))

        # 动态融合注意力
        dynamic_attention = (freq_weight * freq_attention +
                           time_weight * time_attention +
                           static_attention) / 3.0

        return x * dynamic_attention


class RFBoostInceptionDNet(nn.Module):
    """专门为频谱图设计的InceptionTime-Attention模型"""
    def __init__(self, input_shape=(1, 129, 47), num_classes=9, num_modules=2, hidden_channels=32):
        super().__init__()

        # 增强的Stem层，专门处理频谱图
        self.stem = nn.Sequential(
            # 第一层：捕获局部时频特征
            nn.Conv2d(1, hidden_channels//2, kernel_size=(7, 7), padding='same', bias=False),
            nn.BatchNorm2d(hidden_channels//2),
            nn.GELU(),

            # 第二层：进一步提取特征
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=(5, 5), padding='same', bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            # 轻微下采样，保留重要信息
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # 堆叠的频谱图Inception模块
        self.inception_modules = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        current_channels = hidden_channels
        for i in range(num_modules):
            # 使用专门的频谱图Inception模块
            # SpectrogramInceptionModule2D实际输出: 6 * bottleneck_channels
            # bottleneck_channels = hidden_channels//2，所以实际输出 = 6 * (hidden_channels//2) = 3 * hidden_channels
            module_out_channels = hidden_channels * 3  # 修正为实际输出通道数
            self.inception_modules.append(
                SpectrogramInceptionModule2D(current_channels, bottleneck_channels=hidden_channels//2)  # 减少bottleneck通道数
            )

            # 每个模块后添加动态空间注意力
            self.spatial_attentions.append(DynamicSpatialAttention(module_out_channels))
            
            # 残差连接投影
            if current_channels != module_out_channels:
                self.residual_projections.append(
                    nn.Conv2d(current_channels, module_out_channels, kernel_size=1, bias=False)
                )
            else:
                self.residual_projections.append(nn.Identity())
            
            current_channels = module_out_channels
        
        # 全局特征聚合
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 增强的分类头
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 处理不同维度的输入
        if x.dim() == 5:
            # 如果是5维 [batch, 1, 1, H, W]，压缩多余维度
            x = x.squeeze(1).squeeze(1)  # 变成 [batch, H, W]
            x = x.unsqueeze(1)  # 添加通道维度 [batch, 1, H, W]
        elif x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1:
            # 如果是4维但有多余的单维度 [batch, 1, 1, features]
            x = x.squeeze(1).squeeze(1)  # 压缩单维度
            if x.dim() == 2:  # 如果变成2维，需要重塑
                # 假设是频谱图数据，重塑为合适的2D形状
                batch_size = x.shape[0]
                features = x.shape[1]
                # 尝试重塑为接近正方形的形状
                h = int(features ** 0.5)
                w = features // h
                if h * w != features:
                    # 如果不能完美分解，填充到最接近的尺寸
                    target_size = h * (h + 1)
                    if target_size - features <= features - h * h:
                        h = h + 1
                        w = h
                    else:
                        w = h
                    # 填充或截断
                    if h * w > features:
                        padding = h * w - features
                        x = torch.nn.functional.pad(x, (0, padding))
                    else:
                        x = x[:, :h*w]
                x = x.view(batch_size, 1, h, w)
            else:
                x = x.unsqueeze(1)  # 添加通道维度
        elif x.dim() == 4 and x.shape[1:] == (1, 129, 47):
            x = x.squeeze(1).unsqueeze(1)  # 确保正确的通道维度
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        elif x.dim() == 2:
            # 2维输入，需要重塑为2D图像格式
            batch_size, features = x.shape
            # 尝试重塑为接近正方形
            h = int(features ** 0.5)
            w = features // h
            if h * w != features:
                # 填充到最接近的正方形
                target_h = int(features ** 0.5) + 1
                target_w = target_h
                target_features = target_h * target_w
                x = torch.nn.functional.pad(x, (0, target_features - features))
                h, w = target_h, target_w
            x = x.view(batch_size, 1, h, w)
        
        # Stem层处理
        x = self.stem(x)
        
        # 通过Inception模块和注意力机制
        residual = x
        for i, (inception_module, spatial_attention) in enumerate(zip(self.inception_modules, self.spatial_attentions)):
            # Inception特征提取
            x_out = inception_module(x)
            
            # 残差连接
            residual_proj = self.residual_projections[i](residual)
            x = x_out + residual_proj
            
            # 空间注意力
            x = spatial_attention(x)
            
            residual = x
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x





class EMA:
    """指数移动平均，用于稳定模型参数"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}