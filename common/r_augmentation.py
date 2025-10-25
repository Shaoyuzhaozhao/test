"""
RFBoost增强数据增强模块
基于RFBoost论文的核心物理数据增强策略
去除高级物理增强，专注于FDA、TDA、MRE和ISS核心策略
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import random
from scipy.signal import stft
import warnings

class RFBoostAugmenter:
    """
    RFBoost数据增强器
    实现RFBoost论文中的核心物理数据增强策略
    """
    
    def __init__(self, config: Union[dict, object]):
        """
        初始化RFBoost增强器
        
        Args:
            config: 配置对象或配置字典
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = config.to_dict() if hasattr(config, 'to_dict') else config.__dict__
            
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # RFBoost核心参数
        self.fda_k_best = self.config.get("fda_k_best_subcarriers", 6)
        self.tda_window_lengths = self.config.get("tda_window_lengths", [128, 256, 512])
        self.mre_erase_ratio = self.config.get("mre_erase_ratio", 0.2)
        self.default_win_length = self.config.get("default_win_length", 256)
        
        # STFT参数
        self.n_fft = self.config.get("n_fft", 256)
        self.hop_length = self.config.get("hop_length", 64)
        
        # 缓存
        self._subcarrier_stats_cache = {}
    
    def _calculate_subcarrier_motion_stats(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        计算子载波运动统计量
        
        Args:
            csi_data: CSI数据 [batch_size, subcarriers, time_steps] 或 [subcarriers, time_steps]
            
        Returns:
            运动统计量 [subcarriers]
        """
        if csi_data.dim() == 3:
            # 批处理模式，取第一个样本
            csi_data = csi_data[0]
        
        # 计算幅度
        if torch.is_complex(csi_data):
            amplitude = torch.abs(csi_data)
        else:
            amplitude = csi_data
        
        # 计算时间维度的方差作为运动统计量
        motion_stats = torch.var(amplitude, dim=-1)
        
        return motion_stats
    
    def frequency_domain_augmentation(self, csi_data: torch.Tensor, k_best: Optional[int] = None) -> torch.Tensor:
        """
        频域增强 (FDA) - RFBoost核心策略
        选择运动最活跃的k个子载波
        
        Args:
            csi_data: CSI数据，形状可能为 (time_steps,) 或 (subcarriers, time_steps) 或 (batch_size, subcarriers, time_steps)
            k_best: 选择的子载波数量
            
        Returns:
            增强后的CSI数据
        """
        if k_best is None:
            k_best = self.fda_k_best
        
        # 处理不同维度的输入
        original_shape = csi_data.shape
        batch_mode = False
        
        if csi_data.dim() == 1:
            # 单个时间序列 (time_steps,) -> (1, 1, time_steps)
            csi_data = csi_data.unsqueeze(0).unsqueeze(0)
            batch_mode = False
        elif csi_data.dim() == 2:
            # 多子载波时间序列 (subcarriers, time_steps) -> (1, subcarriers, time_steps)
            csi_data = csi_data.unsqueeze(0)
            batch_mode = False
        elif csi_data.dim() == 3:
            # 批量数据 (batch_size, subcarriers, time_steps)
            batch_mode = True
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        batch_size, n_subcarriers, time_steps = csi_data.shape
        
        # 确保k_best不超过子载波数量
        k_best = min(k_best, n_subcarriers)
        
        augmented_data = []
        
        for i in range(batch_size):
            # 计算运动统计量
            motion_stats = self._calculate_subcarrier_motion_stats(csi_data[i])
            
            # 选择运动最活跃的k个子载波
            _, top_k_indices = torch.topk(motion_stats, k_best)
            
            # 提取选中的子载波
            selected_data = csi_data[i, top_k_indices, :]
            augmented_data.append(selected_data)
        
        result = torch.stack(augmented_data, dim=0)
        
        # 根据原始输入维度返回相应格式
        if original_shape == 1:
            # 原始输入是1D，返回1D
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            # 原始输入是2D，返回2D
            result = result.squeeze(0)
        # 3D输入保持3D输出
        
        return result
    
    def time_domain_augmentation(self, csi_data: torch.Tensor, window_length: Optional[int] = None) -> torch.Tensor:
        """
        时域增强 (TDA) - RFBoost核心策略
        使用不同窗口长度生成频谱图
        
        Args:
            csi_data: CSI数据 [batch_size, subcarriers, time_steps] 或 [subcarriers, time_steps]
            window_length: STFT窗口长度
            
        Returns:
            频谱图 [batch_size, subcarriers, freq_bins, time_frames] 或 [subcarriers, freq_bins, time_frames]
        """
        if window_length is None:
            window_length = random.choice(self.tda_window_lengths)
        
        original_shape = csi_data.shape
        batch_mode = csi_data.dim() == 3
        
        if not batch_mode:
            csi_data = csi_data.unsqueeze(0)
        
        batch_size, n_subcarriers, time_steps = csi_data.shape
        
        spectrograms = []
        
        for i in range(batch_size):
            subcarrier_spectrograms = []
            
            for j in range(n_subcarriers):
                # 转换为numpy进行STFT
                signal = csi_data[i, j].cpu().numpy()
                
                if np.iscomplexobj(signal):
                    signal = np.abs(signal)
                
                # 执行STFT
                try:
                    # 确保n_fft >= nperseg (window_length)
                    effective_nfft = max(self.n_fft, window_length)
                    
                    _, _, Zxx = stft(signal, 
                                   nperseg=window_length, 
                                   noverlap=window_length//2,
                                   nfft=effective_nfft)
                    
                    # 转换为幅度谱
                    magnitude = np.abs(Zxx)
                    
                    # 转换回tensor
                    magnitude_tensor = torch.from_numpy(magnitude).float().to(self.device)
                    subcarrier_spectrograms.append(magnitude_tensor)
                    
                except Exception as e:
                    warnings.warn(f"STFT failed for subcarrier {j}: {e}")
                    # 创建零填充的频谱图
                    effective_nfft = max(self.n_fft, window_length)
                    freq_bins = effective_nfft // 2 + 1
                    time_frames = max(1, time_steps // (window_length // 2))
                    zero_spec = torch.zeros(freq_bins, time_frames).to(self.device)
                    subcarrier_spectrograms.append(zero_spec)
            
            # 堆叠所有子载波的频谱图
            batch_spectrogram = torch.stack(subcarrier_spectrograms, dim=0)
            spectrograms.append(batch_spectrogram)
        
        result = torch.stack(spectrograms, dim=0)
        
        if not batch_mode:
            result = result.squeeze(0)
        
        return result
    
    def motion_random_erasing(self, spectrogram: torch.Tensor, erase_ratio: Optional[float] = None) -> torch.Tensor:
        """
        运动随机擦除 (MRE) - RFBoost核心策略
        基于运动统计量随机擦除频谱图区域
        
        Args:
            spectrogram: 频谱图 [batch_size, subcarriers, freq_bins, time_frames] 或 [subcarriers, freq_bins, time_frames]
            erase_ratio: 擦除比例
            
        Returns:
            擦除后的频谱图
        """
        if erase_ratio is None:
            erase_ratio = self.mre_erase_ratio
        
        if random.random() > erase_ratio:
            return spectrogram
        
        original_shape = spectrogram.shape
        batch_mode = spectrogram.dim() == 4
        
        if not batch_mode:
            spectrogram = spectrogram.unsqueeze(0)
        
        batch_size, n_subcarriers, freq_bins, time_frames = spectrogram.shape
        
        erased_spectrograms = []
        
        for i in range(batch_size):
            erased_spec = spectrogram[i].clone()
            
            # 计算每个子载波的运动强度
            motion_intensity = torch.mean(erased_spec, dim=(1, 2))  # [subcarriers]
            
            # 选择运动强度较低的区域进行擦除
            _, low_motion_indices = torch.topk(motion_intensity, 
                                             k=max(1, n_subcarriers // 4), 
                                             largest=False)
            
            for subcarrier_idx in low_motion_indices:
                # 随机选择擦除区域
                erase_h = random.randint(1, freq_bins // 3)
                erase_w = random.randint(1, time_frames // 3)
                
                start_h = random.randint(0, freq_bins - erase_h)
                start_w = random.randint(0, time_frames - erase_w)
                
                # 擦除区域（设为0）
                erased_spec[subcarrier_idx, start_h:start_h+erase_h, start_w:start_w+erase_w] = 0
            
            erased_spectrograms.append(erased_spec)
        
        result = torch.stack(erased_spectrograms, dim=0)
        
        if not batch_mode:
            result = result.squeeze(0)
        
        return result
    
    def intelligent_subcarrier_selection(self, csi_data: torch.Tensor, n_selected: Optional[int] = None) -> torch.Tensor:
        """
        智能子载波选择 (ISS) - RFBoost扩展策略
        基于信号质量和运动特征选择子载波
        
        Args:
            csi_data: CSI数据，形状可能为 (time_steps,) 或 (n_subcarriers, time_steps) 或 (batch_size, n_subcarriers, time_steps)
            n_selected: 选择的子载波数量，如果为None则使用配置中的rfboost_iss_top_k
            
        Returns:
            选择后的CSI数据
        """
        original_shape = csi_data.shape
        batch_mode = False
        
        if csi_data.dim() == 1:
            # 单个时间序列 (time_steps,) -> (1, 1, time_steps)
            csi_data = csi_data.unsqueeze(0).unsqueeze(0)
            batch_mode = False
        elif csi_data.dim() == 2:
            # 多子载波时间序列 (n_subcarriers, time_steps) -> (1, n_subcarriers, time_steps)
            csi_data = csi_data.unsqueeze(0)
            batch_mode = False
        elif csi_data.dim() == 3:
            # 批量数据 (batch_size, n_subcarriers, time_steps)
            batch_mode = True
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        batch_size, n_subcarriers, time_steps = csi_data.shape
        
        # 使用配置中的rfboost_iss_top_k参数
        if n_selected is None:
            n_selected = self.config.get("rfboost_iss_top_k", 10)
        
        # 确保选择数量不超过总子载波数量
        n_selected = min(n_selected, n_subcarriers)
        
        selected_data = []
        
        for i in range(batch_size):
            # 计算信号质量指标
            if torch.is_complex(csi_data[i]):
                amplitude = torch.abs(csi_data[i])
            else:
                amplitude = csi_data[i]
            
            # 信号强度
            signal_power = torch.mean(amplitude ** 2, dim=-1)
            
            # 运动变化
            motion_variance = torch.var(amplitude, dim=-1)
            
            # 信噪比估计
            signal_mean = torch.mean(amplitude, dim=-1)
            noise_estimate = torch.std(amplitude, dim=-1)
            snr = signal_mean / (noise_estimate + 1e-8)
            
            # 综合评分
            quality_score = 0.4 * signal_power + 0.4 * motion_variance + 0.2 * snr
            
            # 选择最佳子载波
            _, selected_indices = torch.topk(quality_score, n_selected)
            selected_indices = torch.sort(selected_indices)[0]  # 保持顺序
            
            selected_subcarriers = csi_data[i, selected_indices, :]
            selected_data.append(selected_subcarriers)
        
        result = torch.stack(selected_data, dim=0)
        
        # 根据原始输入维度返回相应格式
        if len(original_shape) == 1:
            # 原始输入是1D，返回1D
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 2:
            # 原始输入是2D，返回2D
            result = result.squeeze(0)
        # 3D输入保持3D输出
        
        return result
    
    def normalize_spectrogram(self, spectrogram: torch.Tensor, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        频谱图尺寸归一化
        
        Args:
            spectrogram: 频谱图
            target_size: 目标尺寸 (height, width)
            
        Returns:
            归一化后的频谱图
        """
        if spectrogram.dim() == 4:
            # [batch_size, subcarriers, freq_bins, time_frames]
            batch_size, n_subcarriers, freq_bins, time_frames = spectrogram.shape
            
            # 合并子载波维度到通道维度或进行平均
            if n_subcarriers <= 3:
                # 如果子载波数量少，直接作为通道
                combined = spectrogram
            else:
                # 否则进行平均或选择前3个
                combined = spectrogram[:, :3, :, :] if n_subcarriers >= 3 else spectrogram
            
            # 调整到目标尺寸
            resized = F.interpolate(combined, size=target_size, mode='bilinear', align_corners=False)
            
        elif spectrogram.dim() == 3:
            # [subcarriers, freq_bins, time_frames]
            n_subcarriers, freq_bins, time_frames = spectrogram.shape
            
            # 添加批次维度
            spectrogram = spectrogram.unsqueeze(0)
            
            # 处理子载波
            if n_subcarriers <= 3:
                combined = spectrogram
            else:
                combined = spectrogram[:, :3, :, :] if n_subcarriers >= 3 else spectrogram
            
            # 调整到目标尺寸
            resized = F.interpolate(combined, size=target_size, mode='bilinear', align_corners=False)
            resized = resized.squeeze(0)
            
        else:
            raise ValueError(f"Unsupported spectrogram dimension: {spectrogram.dim()}")
        
        return resized
    
    def __call__(self, csi_data: torch.Tensor, augment_type: str = "fda_tda_mre", **kwargs) -> torch.Tensor:
        """
        执行数据增强
        
        Args:
            csi_data: 输入CSI数据
            augment_type: 增强类型，支持 "fda", "tda", "mre", "iss" 及其组合
            **kwargs: 额外参数
            
        Returns:
            增强后的数据
        """
        # 确保数据在正确的设备上
        if isinstance(csi_data, torch.Tensor):
            csi_data = csi_data.to(self.device)
        
        result = csi_data
        
        # 解析增强类型
        augment_types = augment_type.lower().split('_')
        
        # 应用频域增强
        if 'fda' in augment_types:
            result = self.frequency_domain_augmentation(result, **kwargs)
        
        # 应用智能子载波选择
        if 'iss' in augment_types:
            result = self.intelligent_subcarrier_selection(result, **kwargs)
        
        # 应用时域增强
        if 'tda' in augment_types:
            result = self.time_domain_augmentation(result, **kwargs)
        
        # 应用运动随机擦除
        if 'mre' in augment_types and result.dim() >= 3:
            result = self.motion_random_erasing(result, **kwargs)
        
        # 尺寸归一化
        if kwargs.get('normalize', True) and result.dim() >= 3:
            target_size = kwargs.get('target_size', (224, 224))
            result = self.normalize_spectrogram(result, target_size)
        
        return result


class RFBoostAugmentationStrategy:
    """
    RFBoost增强策略管理器
    """
    
    def __init__(self, config: Union[dict, object]):
        """
        初始化增强策略
        
        Args:
            config: 配置对象或字典
        """
        self.config = config if isinstance(config, dict) else config.to_dict()
        self.augmenter = RFBoostAugmenter(config)
        
        # 获取当前策略
        self.strategy = self.config.get("rfboost_strategy", "full")
        self.strategy_type = self.get_strategy(self.strategy)
    
    def get_strategy(self, strategy_name: str) -> str:
        """
        获取增强策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            增强类型字符串
        """
        strategies = {
            'fda_only': 'fda',
            'tda_only': 'tda', 
            'mre_only': 'mre',
            'iss_only': 'iss',
            'fda_tda': 'fda_tda',
            'fda_mre': 'fda_mre',
            'tda_mre': 'tda_mre',
            'fda_tda_mre': 'fda_tda_mre',
            'iss_fda': 'iss_fda',
            'iss_tda': 'iss_tda',
            'iss_mre': 'iss_mre',
            'iss_fda_tda': 'iss_fda_tda',
            'iss_fda_mre': 'iss_fda_mre',
            'iss_tda_mre': 'iss_tda_mre',
            'full': 'iss_fda_tda_mre',
            'none': ''
        }
        
        return strategies.get(strategy_name, strategy_name)
    
    def should_use_fda(self) -> bool:
        """是否应该使用频域增强(FDA)"""
        return 'fda' in self.strategy_type
    
    def should_use_tda(self) -> bool:
        """是否应该使用时域增强(TDA)"""
        return 'tda' in self.strategy_type
    
    def should_use_mre(self) -> bool:
        """是否应该使用运动随机擦除(MRE)"""
        return 'mre' in self.strategy_type
    
    def should_use_iss(self) -> bool:
        """是否应该使用智能子载波选择(ISS)"""
        return 'iss' in self.strategy_type
    
    def get_fda_variants(self) -> int:
        """获取FDA变体数量"""
        return self.config.get("fda_k_best_subcarriers", 15) if self.should_use_fda() else 1
    
    def get_tda_variants(self) -> int:
        """获取TDA变体数量"""
        if self.should_use_tda():
            tda_windows = self.config.get("tda_window_lengths", [128, 256, 512])
            return len(tda_windows)
        return 1
    
    def get_mre_variants(self) -> int:
        """获取MRE变体数量"""
        return 2 if self.should_use_mre() else 1
    
    def get_total_variants(self) -> int:
        """获取总变体数量"""
        return self.get_fda_variants() * self.get_tda_variants() * self.get_mre_variants()
    
    def get_window_lengths(self) -> List[int]:
        """获取所有窗口长度"""
        if self.should_use_tda():
            tda_windows = self.config.get("tda_window_lengths", [128, 256, 512])
            return tda_windows
        # 非TDA时，返回默认窗口长度，用于基础频谱转换
        return [self.default_win_length]
    
    def apply_augmentation(self, csi_data: torch.Tensor, strategy_name: str = "fda_tda_mre", **kwargs) -> torch.Tensor:
        """
        应用增强策略
        
        Args:
            csi_data: CSI数据
            strategy_name: 策略名称
            **kwargs: 额外参数
            
        Returns:
            增强后的数据
        """
        augment_type = self.get_strategy(strategy_name)
        
        if not augment_type:
            return csi_data
        
        return self.augmenter(csi_data, augment_type, **kwargs)


# 便捷函数
def create_rfboost_augmenter(config: Union[dict, object]) -> RFBoostAugmenter:
    """
    创建RFBoost增强器
    
    Args:
        config: 配置对象或字典
        
    Returns:
        RFBoost增强器实例
    """
    return RFBoostAugmenter(config)


def create_rfboost_strategy(config: Union[dict, object]) -> RFBoostAugmentationStrategy:
    """
    创建RFBoost增强策略
    
    Args:
        config: 配置对象或字典
        
    Returns:
        RFBoost增强策略实例
    """
    return RFBoostAugmentationStrategy(config)


def create_rfboost_augmentation_strategy(config: Union[dict, object]) -> RFBoostAugmentationStrategy:
    """
    创建RFBoost增强策略（兼容性函数）
    
    Args:
        config: 配置对象或字典
        
    Returns:
        RFBoost增强策略实例
    """
    return RFBoostAugmentationStrategy(config)


# 兼容性函数
def get_rfboost_augmentation_config() -> dict:
    """
    获取RFBoost增强的默认配置
    
    Returns:
        默认配置字典
    """
    return {
        # RFBoost核心参数
        "fda_k_best_subcarriers": 6,
        "tda_window_lengths": [128, 256, 512],
        "mre_erase_ratio": 0.2,
        "default_win_length": 256,
        
        # STFT参数
        "n_fft": 256,
        "hop_length": 64,
        
        # 设备配置
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # 增强策略
        "augmentation_strategy": "fda_tda_mre",
        "normalize_output": True,
        "target_size": (224, 224)
    }