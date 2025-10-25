"""
实验管理模块
包含实验运行器、结果管理等实验相关功能
"""

import os
import gc
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from .config import Config
from .dataset import WiMANDataset, create_dataset
from .training import create_trainer
from .utils import set_seed


class ExperimentRunner:
    """实验运行器 - 管理完整的实验流程"""
    
    def __init__(self, config, model_class, model_name="Model"):
        """
        初始化实验运行器
        
        Args:
            config: 配置对象或字典
            model_class: 模型类
            model_name: 模型名称，用于输出目录和结果标识
        """
        if isinstance(config, dict):
            self.config = Config(config)
        else:
            self.config = config
            
        self.model_class = model_class
        self.model_name = model_name
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.df = self._prepare_dataframe()
        
        # 设置输出目录
        self._setup_output_directory()

    def _setup_output_directory(self):
        """设置输出目录"""
        # 检查数据增强配置
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        if not enable_augmentation:
            strategy = "no_augmentation"
        elif use_rfboost:
            strategy = "rfboost"
        else:
            strategy = self.config.get("augment_strategy", "fda_tda_mre")
            
        task = self.config.get("task", "activity")
        frequency_band = self.config.get("frequency_band", "2.4")
        environment = self.config.get("environment", "classroom")
        use_preprocessing = self.config.get("use_preprocessing", False)
        preprocessing_suffix = "_use_preprocessing" if use_preprocessing else "_no_preprocessing"
        
        # 使用模型名称作为基础输出目录
        base_output_dir = self.model_name.lower().replace("-", "_")
        
        output_dir = f"{base_output_dir}_{strategy}_{task}_{environment}_{frequency_band}G{preprocessing_suffix}_results"
        self.config.set("output_dir", output_dir)
        os.makedirs(output_dir, exist_ok=True)

    def _prepare_dataframe(self):
        """准备数据框架"""
        annotation_path = os.path.join(self.config.get("data_dir"), self.config.get("annotation_file"))
        df = pd.read_csv(annotation_path)
        
        # 频段筛选
        frequency_column = self.config.get("frequency_column_name")
        df[frequency_column] = df[frequency_column].astype(str)
        frequency_band = self.config.get("frequency_band")
        df = df[df[frequency_column].str.contains(frequency_band, na=False, case=False)]
        
        # 环境筛选
        environment_column = self.config.get("environment_column_name")
        environment = self.config.get("environment")
        df = df[df[environment_column].str.lower() == environment.lower()]

        # 添加完整路径
        amp_dir = os.path.join(self.config.get("data_dir"), self.config.get("csi_amp_dir"))
        sample_id_column = self.config.get("sample_id_column")
        df['full_path_amp'] = df[sample_id_column].apply(lambda x: os.path.join(amp_dir, f"{x}.npy"))

        # 过滤存在的文件
        df = df[df['full_path_amp'].apply(os.path.exists)].reset_index(drop=True)
        
        print(f"找到 {len(df)} 个有效样本 (环境: '{environment}', 频段: '{frequency_band}G').")
        return df

    def _create_data_loaders(self, train_df, val_df):
        """创建数据加载器"""
        train_dataset = create_dataset(train_df, self.config.to_dict(), is_train=True)
        val_dataset = create_dataset(val_df, self.config.to_dict(), is_train=False)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get("batch_size"), 
            shuffle=True,
            num_workers=self.config.get("num_workers"), 
            pin_memory=self.config.get("pin_memory", True),
            prefetch_factor=self.config.get("prefetch_factor", 2),
            persistent_workers=self.config.get("persistent_workers", False)
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.get("batch_size"), 
            shuffle=False,
            num_workers=self.config.get("num_workers"), 
            pin_memory=self.config.get("pin_memory", True),
            prefetch_factor=self.config.get("prefetch_factor", 2),
            persistent_workers=self.config.get("persistent_workers", False)
        )
        
        return train_loader, val_loader, train_dataset.num_classes

    def _split_data(self, random_seed):
        """分割数据集"""
        set_seed(random_seed)
        
        train_size = int(0.8 * len(self.df))
        val_size = len(self.df) - train_size
        train_indices, val_indices = random_split(range(len(self.df)), [train_size, val_size])

        train_df = self.df.iloc[train_indices.indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices.indices].reset_index(drop=True)
        
        return train_df, val_df

    def run_single_experiment(self, task, repeat_idx):
        """运行单次实验"""
        print(f"\n--- 第 {repeat_idx}/{self.config.get('repeats')} 次重复 ---")
        
        # 分割数据
        random_seed = self.config.get("seed") + repeat_idx
        train_df, val_df = self._split_data(random_seed)
        
        # 创建数据加载器
        train_loader, val_loader, num_classes = self._create_data_loaders(train_df, val_df)
        
        # 创建模型
        model = self.model_class(num_classes=num_classes)
        
        # 创建训练器
        config_dict = self.config.to_dict()
        config_dict["model_name"] = self.model_name  # 添加模型名称到配置中
        trainer = create_trainer(model, self.device, task, repeat_idx, config_dict)
        
        # 训练模型
        best_acc = trainer.train(train_loader, val_loader)
        
        # 清理内存
        gc.collect()
        if self.device.type == 'cuda': 
            torch.cuda.empty_cache()
            
        return best_acc

    def run(self):
        """运行完整实验"""
        all_task_results = {}
        task = self.config.get('task')
        
        print(f"\n{'=' * 50}\n开始实验任务: {task.upper()}\n{'=' * 50}")
        task_results = []

        # 检查数据集是否为空
        full_dataset_for_split = create_dataset(self.df, self.config.to_dict(), is_train=True)
        if len(full_dataset_for_split) == 0: 
            print(f"任务 {task} 没有数据，跳过。")
            return {}

        # 运行多次重复实验
        repeats = self.config.get("repeats", 1)
        for r_idx in range(1, repeats + 1):
            best_acc = self.run_single_experiment(task, r_idx)
            task_results.append(best_acc)

        # 计算统计结果
        avg_acc, std_acc = (np.mean(task_results), np.std(task_results)) if task_results else (0.0, 0.0)
        print(f"\n任务 '{task}' 完成. 准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        
        # 显示每次重复的详细结果
        if task_results:
            print(f"详细结果: {[f'{acc:.4f}' for acc in task_results]}")
            print(f"最高准确率: {max(task_results):.4f}")
            print(f"最低准确率: {min(task_results):.4f}")
        
        all_task_results[task] = {"mean": avg_acc, "std": std_acc, "all_results": task_results}

        # 保存结果
        self.save_results(all_task_results)
        
        return all_task_results

    def save_results(self, all_task_results):
        """保存实验结果"""
        # 检查数据增强配置
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        if not enable_augmentation:
            augment_strategy = "no_augmentation"
        elif use_rfboost:
            augment_strategy = "rfboost"
        else:
            augment_strategy = self.config.get("augment_strategy")
            
        results_data = {
            "model": self.model_name,
            "augment_strategy": augment_strategy,
            "config": {k: v for k, v in self.config.to_dict().items() if k not in ["device"]},
            "results": all_task_results
        }
        
        task = self.config.get('task')
        environment = self.config.get('environment')
        frequency_band = self.config.get('frequency_band')
        
        results_path = os.path.join(
            self.config.get("output_dir"),
            f"results_{task}_{environment}_{frequency_band}G.json"
        )
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        print(f"\n结果已保存至: {results_path}")


class ResultsManager:
    """结果管理器 - 管理和分析实验结果"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
    
    def load_results(self, results_file):
        """加载实验结果"""
        results_path = os.path.join(self.results_dir, results_file)
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_results(self, results_files):
        """比较多个实验结果"""
        all_results = {}
        for file in results_files:
            results = self.load_results(file)
            model_name = results.get("model", "Unknown")
            strategy = results.get("augment_strategy", "Unknown")
            key = f"{model_name}_{strategy}"
            all_results[key] = results
        
        return all_results
    
    def generate_summary(self, results_files):
        """生成结果摘要"""
        comparison = self.compare_results(results_files)
        
        summary = {
            "total_experiments": len(comparison),
            "results": {}
        }
        
        for key, results in comparison.items():
            for task, task_results in results["results"].items():
                if task not in summary["results"]:
                    summary["results"][task] = {}
                summary["results"][task][key] = {
                    "mean": task_results["mean"],
                    "std": task_results["std"]
                }
        
        return summary


# 便捷函数
def create_experiment_runner(config, model_class, model_name="Model"):
    """创建实验运行器的便捷函数"""
    return ExperimentRunner(config, model_class, model_name)


def create_results_manager(results_dir):
    """创建结果管理器的便捷函数"""
    return ResultsManager(results_dir)


def run_experiment(config, model_class, model_name="Model"):
    """运行实验的便捷函数"""
    runner = create_experiment_runner(config, model_class, model_name)
    return runner.run()