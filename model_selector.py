"""
模型选择器模块
支持单个模型选择和批量模型运行
"""
import torch
from models import (
    SpectrogramInceptionTimeAttention2D,
    AlexNet,
    ResNetTS,
    ResNet18,
    RFNet
)

# 模型映射字典
MODEL_REGISTRY = {
    'inception_time_attention': {
        'class': SpectrogramInceptionTimeAttention2D,
        'name': 'InceptionTime-Attention',
        'description': '基于Inception和注意力机制的频谱图模型'
    },
    'alexnet': {
        'class': AlexNet,
        'name': 'AlexNet',
        'description': '经典AlexNet架构适配WiFi感知任务'
    },
    'resnet_ts': {
        'class': ResNetTS,
        'name': 'ResNet-TS',
        'description': '时间序列专用的ResNet模型'
    },
    'resnet18': {
        'class': ResNet18,
        'name': 'ResNet18',
        'description': '标准ResNet18架构'
    },
    'rfnet': {
        'class': RFNet,
        'name': 'RF-Net',
        'description': '专为WiFi感知设计的网络架构'
    }
}

class ModelSelector:
    """模型选择器类"""
    
    def __init__(self, input_shape=(1, 129, 47), num_classes=9):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def get_model(self, model_name):
        """获取指定模型实例"""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"未知模型: {model_name}. 可用模型: {list(MODEL_REGISTRY.keys())}")
        
        model_info = MODEL_REGISTRY[model_name]
        model_class = model_info['class']
        
        return model_class(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
    
    def get_all_models(self):
        """获取所有模型实例的字典"""
        models = {}
        for model_name in MODEL_REGISTRY:
            models[model_name] = self.get_model(model_name)
        return models
    
    def get_model_info(self, model_name=None):
        """获取模型信息"""
        if model_name is None:
            return MODEL_REGISTRY
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"未知模型: {model_name}")
        
        return MODEL_REGISTRY[model_name]
    
    def list_available_models(self):
        """列出所有可用模型"""
        print("可用模型列表:")
        print("-" * 50)
        for model_name, info in MODEL_REGISTRY.items():
            print(f"模型名称: {model_name}")
            print(f"显示名称: {info['name']}")
            print(f"描述: {info['description']}")
            print("-" * 50)

def create_model(model_name, input_shape=(1, 129, 47), num_classes=9):
    """便捷函数：创建指定模型"""
    selector = ModelSelector(input_shape, num_classes)
    return selector.get_model(model_name)

def create_all_models(input_shape=(1, 129, 47), num_classes=9):
    """便捷函数：创建所有模型"""
    selector = ModelSelector(input_shape, num_classes)
    return selector.get_all_models()