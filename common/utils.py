"""
工具函数模块
包含随机种子设置、设备管理、数据处理等通用工具函数
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
import time
from typing import Union, Dict, Any, List, Optional, Tuple
import logging


def set_seed(seed: int):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        device: 指定设备 ('cpu', 'cuda', 'cuda:0' 等)，None时自动选择
        
    Returns:
        torch.device: 计算设备
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        Dict: 包含总参数数、可训练参数数、不可训练参数数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def save_json(data: Dict[str, Any], filepath: str, indent: int = 4):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        indent: JSON缩进
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        Dict: 加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(directory: str):
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
    """
    os.makedirs(directory, exist_ok=True)


def get_memory_usage() -> Dict[str, float]:
    """
    获取内存使用情况
    
    Returns:
        Dict: 内存使用信息（MB）
    """
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    result = {
        "rss": memory_info.rss / 1024 / 1024,  # 物理内存
        "vms": memory_info.vms / 1024 / 1024,  # 虚拟内存
    }
    
    # GPU内存使用情况
    if torch.cuda.is_available():
        result["gpu_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["gpu_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return result


def clear_memory():
    """清理内存"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    验证配置是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        bool: 是否通过验证
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"配置缺少必需的键: {missing_keys}")
    return True


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    设置日志记录
    
    Args:
        log_file: 日志文件路径，None时只输出到控制台
        level: 日志级别
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测值
        targets: 真实标签
        threshold: 二分类阈值
        
    Returns:
        float: 准确率
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        # 多分类
        pred_labels = torch.argmax(predictions, dim=1)
        target_labels = torch.argmax(targets, dim=1) if targets.dim() > 1 else targets
    else:
        # 二分类
        pred_labels = (torch.sigmoid(predictions) > threshold).float()
        target_labels = targets
    
    correct = (pred_labels == target_labels).sum().item()
    total = target_labels.numel()
    
    return correct / total if total > 0 else 0.0


def normalize_tensor(tensor: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    标准化张量
    
    Args:
        tensor: 输入张量
        dim: 标准化的维度，None时对整个张量标准化
        
    Returns:
        torch.Tensor: 标准化后的张量
    """
    if dim is None:
        mean = tensor.mean()
        std = tensor.std()
    else:
        mean = tensor.mean(dim=dim, keepdim=True)
        std = tensor.std(dim=dim, keepdim=True)
    
    return (tensor - mean) / (std + 1e-8)


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...]):
    """
    打印模型摘要
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸 (不包括batch维度)
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # 创建摘要字典
    summary = {}
    hooks = []
    
    # 注册钩子
    model.apply(register_hook)
    
    # 创建输入张量
    device = next(model.parameters()).device
    x = torch.randn(1, *input_size).to(device)
    
    # 前向传播
    model(x)
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
    # 打印摘要
    print("=" * 70)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 70)
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        line_new = f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:>15,}"
        total_params += summary[layer]["nb_params"]
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        
        print(line_new)
    
    print("=" * 70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 70)


class Timer:
    """计时器工具类"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """停止计时"""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """获取经过的时间（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_str(self) -> str:
        """获取格式化的经过时间"""
        return format_time(self.elapsed())
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()


# 便捷函数
def create_timer():
    """创建计时器的便捷函数"""
    return Timer()


def print_separator(title: str = "", char: str = "=", length: int = 50):
    """打印分隔符"""
    if title:
        title_len = len(title)
        if title_len >= length - 4:
            print(char * length)
            print(title)
            print(char * length)
        else:
            padding = (length - title_len - 2) // 2
            print(char * padding + f" {title} " + char * (length - padding - title_len - 2))
    else:
        print(char * length)