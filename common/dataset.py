"""
数据集模块 - 统一管理数据加载和预处理逻辑
"""

import numpy as np
import torch
import pandas as pd
import pywt
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Any, Union, Optional, Tuple, List
from .config import Config
from .augmentation import OnTheFlyAugmenter, AugmentationStrategy
from .r_augmentation import RFBoostAugmenter, RFBoostAugmentationStrategy


class WiMANDataset(Dataset):
    """
    WiFi感知数据集类
    支持多种增强策略、子载波选择、运动统计计算等功能
    """
    
    def __init__(self, annotation_df: pd.DataFrame, config: Union[Config, dict], is_train: bool = True):
        """
        初始化数据集
        
        Args:
            annotation_df: 标注数据框
            config: 配置对象或字典
            is_train: 是否为训练集
        """
        self.config = config if isinstance(config, dict) else config.to_dict()
        self.is_train = is_train
        self.df = annotation_df
        
        # 初始化增强器和策略
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        if enable_augmentation and use_rfboost:
            # 使用RFBoost增强器
            self.on_the_fly_augmenter = RFBoostAugmenter(self.config)
            self.augmentation_strategy = RFBoostAugmentationStrategy(self.config)
            # 获取消融实验策略
            self.rfboost_strategy = self.config.get("rfboost_strategy", "full")
        elif enable_augmentation:
            # 使用原始增强器
            self.on_the_fly_augmenter = OnTheFlyAugmenter(self.config)
            self.augmentation_strategy = AugmentationStrategy(
                self.config.get("augment_strategy", "fda_tda_mre"), 
                self.config
            )
        else:
            # 不使用数据增强，仅基础频谱转换
            self.on_the_fly_augmenter = RFBoostAugmenter(self.config)
            self.augmentation_strategy = RFBoostAugmentationStrategy(self.config)
            # 强制设置为none策略，仅进行基础频谱转换
            self.rfboost_strategy = "none"
        
        self.device = torch.device(self.config["device"])
        
        # GPU优化配置
        self.use_gpu_stats = self.config.get("use_gpu_statistics", False) and torch.cuda.is_available()
        self.use_gpu_augment = self.config.get("use_gpu_augmentation", False) and torch.cuda.is_available()
        
        # 活动标签映射
        self.activity_map = {
            'nothing': 0, 'walk': 1, 'rotation': 2, 'jump': 3, 'wave': 4, 
            'lie_down': 5, 'pick_up': 6, 'sit_down': 7, 'stand_up': 8
        }
        
        # 任务配置
        task_config = self.get_task_config()
        self.num_classes = task_config["num_classes"]
        
        # FDA预计算
        self.use_fda = self.augmentation_strategy.should_use_fda()
        if self.use_fda:
            self._precompute_best_subcarriers()

    def get_task_config(self) -> Dict[str, Any]:
        """
        获取任务配置
        
        Returns:
            任务配置字典
        """
        task = self.config["task"]
        if task == "activity": 
            return {
                "num_classes": 9, 
                "label_cols": [f"user_{i}_activity" for i in range(1, 6)],
                "map": self.activity_map
            }
        raise ValueError(f"Unknown task: {task}")

    def __len__(self) -> int:
        """获取数据集长度"""
        if not self.is_train:
            return len(self.df)
        
        # 训练集考虑增强变体
        total_variants_per_sample = self.augmentation_strategy.get_total_variants()
        return len(self.df) * total_variants_per_sample

    def _denoise_signal(self, signal_row: np.ndarray) -> np.ndarray:
        """
        使用小波变换去噪信号
        
        Args:
            signal_row: 输入信号
            
        Returns:
            去噪后的信号
        """
        coeffs = pywt.wavedec(signal_row, 'db4', level=5)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        if sigma < 1e-8: 
            sigma = 1e-8
        threshold = sigma * np.sqrt(2 * np.log(len(signal_row)))
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(denoised_coeffs, 'db4')[:len(signal_row)]

    def _calculate_motion_statistics(self, csi_data_1d: np.ndarray) -> float:
        """
        计算运动统计量(CPU版本)
        
        Args:
            csi_data_1d: 一维CSI数据
            
        Returns:
            运动统计量
        """
        mean_csi = np.mean(csi_data_1d)
        csi_centered = csi_data_1d - mean_csi
        cov_lag1 = np.sum(csi_centered[:-1] * csi_centered[1:])
        var = np.sum(csi_centered ** 2)
        return np.abs(cov_lag1 / (var + 1e-8))

    def _calculate_motion_statistics_gpu(self, csi_tensor: torch.Tensor) -> torch.Tensor:
        """
        计算运动统计量(GPU版本)
        
        Args:
            csi_tensor: CSI张量
            
        Returns:
            运动统计量张量
        """
        if not isinstance(csi_tensor, torch.Tensor):
            csi_tensor = torch.tensor(csi_tensor, dtype=torch.float32, device=self.device)
        elif csi_tensor.device != self.device:
            csi_tensor = csi_tensor.to(self.device)
            
        mean_csi = torch.mean(csi_tensor)
        csi_centered = csi_tensor - mean_csi
        cov_lag1 = torch.sum(csi_centered[:-1] * csi_centered[1:])
        var = torch.sum(csi_centered ** 2)
        return torch.abs(cov_lag1 / (var + 1e-8))

    def _calculate_motion_statistics_batch(self, csi_data: np.ndarray) -> np.ndarray:
        """
        批量计算运动统计量
        
        Args:
            csi_data: CSI数据
            
        Returns:
            运动统计量数组
        """
        if self.use_gpu_stats:
            if not isinstance(csi_data, torch.Tensor):
                csi_tensor = torch.tensor(csi_data, dtype=torch.float32, device=self.device)
            else:
                csi_tensor = csi_data.to(self.device)
            
            mean_csi = torch.mean(csi_tensor, dim=0, keepdim=True)
            csi_centered = csi_tensor - mean_csi
            cov_lag1 = torch.sum(csi_centered[:-1] * csi_centered[1:], dim=0)
            var = torch.sum(csi_centered ** 2, dim=0)
            motion_stats = torch.abs(cov_lag1 / (var + 1e-8))
            return motion_stats.cpu().numpy()
        else:
            return np.apply_along_axis(self._calculate_motion_statistics, 0, csi_data)

    def _precompute_best_subcarriers(self):
        """预计算FDA的最佳子载波"""
        self.best_subcarrier_indices = []
        print("Pre-computing best subcarriers for FDA...")
        
        for idx in tqdm(range(len(self.df)), desc="FDA Pre-computation"):
            row = self.df.iloc[idx]
            amp_path = row['full_path_amp']
            
            try:
                csi_amp = np.load(amp_path)
                if csi_amp.shape[0] != 3000:
                    csi_amp = np.pad(csi_amp, ((0, 3000 - csi_amp.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')

                if self.config["use_preprocessing"]:
                    csi_amp = np.apply_along_axis(self._denoise_signal, 0, csi_amp)

                flat_csi = csi_amp.reshape(3000, -1)
                motion_stats = self._calculate_motion_statistics_batch(flat_csi)
                
                k = self.config["fda_k_best_subcarriers"]
                best_indices = np.argsort(motion_stats)[-k:]
                self.best_subcarrier_indices.append(best_indices)
            except Exception:
                self.best_subcarrier_indices.append(None)

    def _load_and_preprocess_csi(self, amp_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载和预处理CSI数据
        
        Args:
            amp_path: CSI幅度文件路径
            
        Returns:
            处理后的CSI数据和运动统计量
        """
        try:
            csi_amp = np.load(amp_path)
            if csi_amp.shape[0] != 3000:
                csi_amp = np.pad(csi_amp, ((0, 3000 - csi_amp.shape[0]), (0, 0), (0, 0), (0, 0)), 'constant')
        except Exception:
            return None, None

        if self.config["use_preprocessing"]:
            csi_amp = np.apply_along_axis(self._denoise_signal, 0, csi_amp)

        flat_csi = csi_amp.reshape(3000, -1)
        motion_stats = self._calculate_motion_statistics_batch(flat_csi)
        
        return flat_csi, motion_stats

    def _get_augmentation_params(self, idx: int) -> Tuple[int, int, int, int]:
        """
        获取增强参数
        
        Args:
            idx: 数据索引
            
        Returns:
            原始索引、FDA索引、TDA索引、MRE索引
        """
        total_variants_per_sample = self.augmentation_strategy.get_total_variants()
        fda_variants = self.augmentation_strategy.get_fda_variants()
        tda_variants = self.augmentation_strategy.get_tda_variants()
        mre_variants = self.augmentation_strategy.get_mre_variants()
        
        original_idx = idx // total_variants_per_sample
        variant_idx = idx % total_variants_per_sample
        
        fda_idx = variant_idx // (tda_variants * mre_variants) if fda_variants > 1 else 0
        remaining_idx = variant_idx % (tda_variants * mre_variants)
        
        tda_idx = remaining_idx // mre_variants if tda_variants > 1 else 0
        mre_idx = remaining_idx % mre_variants if mre_variants > 1 else 0
        
        return original_idx, fda_idx, tda_idx, mre_idx

    def _select_subcarrier(self, flat_csi: np.ndarray, motion_stats: np.ndarray, 
                          original_idx: int, fda_idx: int) -> np.ndarray:
        """
        选择子载波
        
        Args:
            flat_csi: 扁平化的CSI数据
            motion_stats: 运动统计量
            original_idx: 原始索引
            fda_idx: FDA索引
            
        Returns:
            选择的CSI时间序列
        """
        if self.augmentation_strategy.should_use_fda():
            best_indices = self.best_subcarrier_indices[original_idx]
            if best_indices is None:
                return None
            chosen_subcarrier_idx = best_indices[fda_idx]
            return flat_csi[:, chosen_subcarrier_idx]
        else:
            best_subcarrier_idx = np.argmax(motion_stats)
            return flat_csi[:, best_subcarrier_idx]

    def _create_augment_type(self, tda_idx: int, mre_idx: int) -> str:
        """
        创建增强类型字符串
        
        Args:
            tda_idx: TDA索引
            mre_idx: MRE索引
            
        Returns:
            增强类型字符串
        """
        if self.augmentation_strategy.should_use_tda():
            window_lengths = self.augmentation_strategy.get_window_lengths()
            win_len = window_lengths[tda_idx]
        else:
            win_len = self.config["default_win_length"]
        
        if self.augmentation_strategy.should_use_mre():
            apply_mre = (mre_idx == 1)
        else:
            apply_mre = False
        
        return f"win_{win_len}_mre_{int(apply_mre)}"

    def _create_label_tensor(self, row: pd.Series) -> torch.Tensor:
        """
        创建标签张量
        
        Args:
            row: 数据行
            
        Returns:
            标签张量
        """
        task_config = self.get_task_config()
        label = torch.zeros(task_config["num_classes"], dtype=torch.float32)
        
        for col in task_config["label_cols"]:
            val = row.get(col)
            if pd.notna(val) and val in task_config["map"]:
                label[task_config["map"][val]] = 1.0
                
        return label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            频谱图张量和标签张量
        """
        task_config = self.get_task_config()
        
        # 检查是否使用数据增强
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        # 测试集或无增强策略
        if not self.is_train or not enable_augmentation or (not use_rfboost and self.config.get("augment_strategy", "fda_tda_mre") == "none"):
            row = self.df.iloc[idx]
            amp_path = row['full_path_amp']

            flat_csi, motion_stats = self._load_and_preprocess_csi(amp_path)
            if flat_csi is None:
                return torch.zeros(1, 129, 47), torch.zeros(task_config["num_classes"])

# RFBoost增强逻辑
            best_subcarrier_idx = np.argmax(motion_stats)
            csi_series = flat_csi[:, best_subcarrier_idx]
            
            # 应用RFBoost消融实验策略
            final_spectrogram = self.augmentation_strategy.apply_augmentation(
                torch.tensor(csi_series, dtype=torch.float32), 
                strategy_name=self.rfboost_strategy
            )
        else:
            # 训练集增强逻辑
            original_idx, fda_idx, tda_idx, mre_idx = self._get_augmentation_params(idx)
            
            row = self.df.iloc[original_idx]
            amp_path = row['full_path_amp']

            flat_csi, motion_stats = self._load_and_preprocess_csi(amp_path)
            if flat_csi is None:
                return torch.zeros(1, 129, 47), torch.zeros(task_config["num_classes"])
            
            # 选择子载波
            csi_series = self._select_subcarrier(flat_csi, motion_stats, original_idx, fda_idx)
            if csi_series is None:
                return torch.zeros(1, 129, 47), torch.zeros(task_config["num_classes"])
            
            # 创建增强类型并应用RFBoost增强
            if hasattr(self, 'rfboost_strategy'):
                # 使用消融实验策略
                final_spectrogram = self.augmentation_strategy.apply_augmentation(
                    torch.tensor(csi_series, dtype=torch.float32), 
                    strategy_name=self.rfboost_strategy
                )
            else:
                # 使用传统增强方式
                augment_type = self._create_augment_type(tda_idx, mre_idx)
                final_spectrogram = self.on_the_fly_augmenter(csi_series, augment_type, use_gpu=self.use_gpu_augment)

        # 创建标签
        label = self._create_label_tensor(row)

        return final_spectrogram, label


def create_dataset(annotation_df: pd.DataFrame, 
                  config: Union[Config, dict], 
                  is_train: bool = True) -> WiMANDataset:
    """
    创建数据集的便捷函数
    
    Args:
        annotation_df: 标注数据框
        config: 配置对象或字典
        is_train: 是否为训练集
        
    Returns:
        数据集实例
    """
    return WiMANDataset(annotation_df, config, is_train)


def load_annotation_data(config: Union[Config, dict]) -> pd.DataFrame:
    """
    加载标注数据
    
    Args:
        config: 配置对象或字典
        
    Returns:
        标注数据框
    """
    config_dict = config if isinstance(config, dict) else config.to_dict()
    
    annotation_path = os.path.join(config_dict["data_dir"], config_dict["annotation_file"])
    df = pd.read_csv(annotation_path)
    
    # 构建完整路径
    df['full_path_amp'] = df.apply(
        lambda row: os.path.join(
            config_dict["data_dir"], 
            config_dict["csi_amp_dir"], 
            f"{row[config_dict['sample_id_column']]}.npy"
        ), 
        axis=1
    )
    
    return df


def filter_annotation_data(df: pd.DataFrame, config: Union[Config, dict]) -> pd.DataFrame:
    """
    根据配置过滤标注数据
    
    Args:
        df: 原始数据框
        config: 配置对象或字典
        
    Returns:
        过滤后的数据框
    """
    config_dict = config if isinstance(config, dict) else config.to_dict()
    
    # 频段过滤
    if config_dict.get("frequency_band"):
        df = df[df[config_dict["frequency_column_name"]] == config_dict["frequency_band"]]
    
    # 环境过滤
    if config_dict.get("environment"):
        df = df[df[config_dict["environment_column_name"]] == config_dict["environment"]]
    
    return df.reset_index(drop=True)