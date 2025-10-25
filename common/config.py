"""
配置管理模块 - 统一管理所有模型的配置参数
"""

import os
import json
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """配置管理类，支持从字典、文件或环境变量加载配置"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典，如果为None则使用默认配置
        """
        self._config = self._get_default_config()
        if config_dict:
            self._config.update(config_dict)
        
        # 验证配置
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # --- 路径和文件配置 ---
            "data_dir": "./dataset",
            "annotation_file": "annotation.csv",
            "csi_amp_dir": "wifi_csi/amp",

            # --- 标注文件列名配置 ---
            "sample_id_column": "label",
            "frequency_column_name": "wifi_band",
            "environment_column_name": "environment",

            # --- 任务和筛选配置 ---
            "task": "activity",
            "frequency_band": "5",
            "environment": "classroom",
            "use_preprocessing": False,

            # --- 频率增强配置 (FDA) ---
            "fda_k_best_subcarriers": 15,  # 基于270个总子载波调整，约5.6%

            # --- STFT 频谱图转换配置 ---
            "n_fft": 256,
            "hop_length": 64,
            "default_win_length": 256,

            # --- RFBoost 在线增强配置 ---
            "use_rfboost_on_the_fly": True,
            "tda_window_lengths": [128, 512],
            "mre_erase_ratio": 0.2,
            
            # --- 训练超参数 ---
            "batch_size": 32,  # 减小批次大小以节省GPU内存
            "epochs": 20,
            "repeats": 3,
            "learning_rate": 1e-3,
            "learning_rate_warmup_epochs": 0,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "early_stopping_patience": 30,
            "max_grad_norm": 2.0,
            
            # --- 数据增强总配置 ---
            "enable_data_augmentation": True,  # 是否启用数据增强（总开关）
            
            # --- RFBoost增强配置 ---
            "use_rfboost_augmentation": True,  # 是否使用RFBoost增强器（当enable_data_augmentation=True时生效）
            "rfboost_cache_size": 500,  # RFBoost缓存大小（减小以节省内存）
            "rfboost_use_iss": False,  # 暂时关闭智能子载波选择以节省内存和时间
            "rfboost_iss_top_k": 25,  # ISS选择的top-k子载波数量，基于270个总子载波调整，约9.3%
            "rfboost_adaptive_mre": True,  # 是否使用自适应MRE
            "rfboost_mre_threshold": 0.3,  # MRE运动阈值
            
            # --- RFBoost消融实验配置 ---
            "rfboost_strategy": "full",  # 增强策略选择，用于消融实验
            # 可选值: "fda_only", "tda_only", "mre_only", "iss_only", 
            #        "fda_tda", "fda_mre", "tda_mre", "fda_tda_mre",
            #        "iss_fda", "iss_tda", "iss_mre", "iss_fda_tda", 
            #        "iss_fda_mre", "iss_tda_mre", "full", "none"
            
            # --- 稳定性配置 ---
            "use_ema": False,
            "ema_decay": 0.999,
            "validation_frequency": 1,
            "num_workers": 0,  # 设置为0以避免多进程CUDA共享问题
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
            
            # --- GPU优化配置 ---
            "use_gpu_augmentation": True,
            "use_gpu_statistics": True,
            "gpu_batch_processing": True,
            "pin_memory": False,  # 保持False以避免内存问题
            "prefetch_factor": None,  # 设置为None以配合num_workers=0
            "persistent_workers": False,  # 保持False以避免进程问题
        }
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        # 验证路径存在性
        if not os.path.exists(self._config["data_dir"]):
            print(f"Warning: Data directory '{self._config['data_dir']}' does not exist")
        
        # 验证数值范围
        assert self._config["batch_size"] > 0, "batch_size must be positive"
        assert self._config["epochs"] > 0, "epochs must be positive"
        assert self._config["learning_rate"] > 0, "learning_rate must be positive"
        assert 0 <= self._config["mre_erase_ratio"] <= 1, "mre_erase_ratio must be between 0 and 1"
        assert 0 <= self._config["label_smoothing"] <= 1, "label_smoothing must be between 0 and 1"
        
        # 验证设备可用性
        if self._config["device"] == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self._config["device"] = "cpu"
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self._config[key] = value
        self._validate_config()
    
    def update(self, config_dict: Dict[str, Any]):
        """批量更新配置"""
        self._config.update(config_dict)
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()
    
    def save(self, filepath: Union[str, Path]):
        """保存配置到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "WIMAN_") -> 'Config':
        """从环境变量加载配置"""
        config = cls()
        
        # 从环境变量中读取配置
        for key in config._config.keys():
            env_key = f"{prefix}{key.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                # 尝试转换类型
                try:
                    if isinstance(config._config[key], bool):
                        config._config[key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(config._config[key], int):
                        config._config[key] = int(env_value)
                    elif isinstance(config._config[key], float):
                        config._config[key] = float(env_value)
                    elif isinstance(config._config[key], list):
                        config._config[key] = json.loads(env_value)
                    else:
                        config._config[key] = env_value
                except (ValueError, json.JSONDecodeError):
                    print(f"Warning: Failed to parse environment variable {env_key}={env_value}")
        
        return config
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return key in self._config
    
    def __str__(self) -> str:
        """字符串表示"""
        return json.dumps(self._config, indent=2, ensure_ascii=False)


# 创建默认配置实例
def get_default_config():
    """获取默认配置实例"""
    return Config()

DEFAULT_CONFIG = get_default_config()

# 便捷函数
def get_config(config_path: Optional[Union[str, Path]] = None, 
               config_dict: Optional[Dict[str, Any]] = None,
               use_env: bool = False) -> Config:
    """
    获取配置实例
    
    Args:
        config_path: 配置文件路径
        config_dict: 配置字典
        use_env: 是否从环境变量加载
    
    Returns:
        Config实例
    """
    if config_path:
        return Config.load(config_path)
    elif use_env:
        return Config.from_env()
    elif config_dict:
        return Config(config_dict)
    else:
        return Config()


def create_experiment_config(base_config: Optional[Config] = None, **kwargs) -> Config:
    """
    创建实验配置
    
    Args:
        base_config: 基础配置
        **kwargs: 要覆盖的配置参数
    
    Returns:
        实验配置实例
    """
    if base_config is None:
        base_config = DEFAULT_CONFIG
    
    config = Config(base_config.to_dict())
    config.update(kwargs)
    return config