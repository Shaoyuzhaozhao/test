"""
训练工具模块
包含训练器、早停、指数移动平均等训练相关工具
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt


class EMA:
    """指数移动平均"""
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


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: 
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience: 
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose: 
            print(f'Validation loss decreased ({self.val_loss_min:.6f}-->{val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class Trainer:
    """通用训练器"""
    def __init__(self, model, device, task, repeat_idx, config):
        self.model = model.to(device)
        self.device = device
        self.task = task
        self.repeat_idx = repeat_idx
        self.config = config
        
        # 优化器设置
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config["epochs"] - config["learning_rate_warmup_epochs"],
            eta_min=config["learning_rate"] * 0.01
        )
        
        # EMA设置
        self.ema = EMA(model, decay=config.get("ema_decay", 0.999)) if config.get("use_ema", False) else None
        
        # 训练历史记录
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _calculate_pos_weights(self, train_dataset):
        """计算正样本权重用于BCEWithLogitsLoss"""
        print("Calculating pos_weights for BCEWithLogitsLoss...")
        labels = torch.stack(
            [train_dataset[i][1] for i in tqdm(range(len(train_dataset)), desc="_calculate_pos_weights")])
        pos_counts = labels.sum(dim=0)
        neg_counts = len(labels) - pos_counts
        return (neg_counts / (pos_counts + 1e-6)).to(self.device)

    def _calculate_accuracy(self, outputs, labels):
        """计算准确率"""
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return (preds == labels).sum().item() / labels.numel() if labels.numel() > 0 else 0

    def _adjust_learning_rate(self, epoch):
        """调整学习率（warmup阶段）"""
        warmup_epochs = self.config["learning_rate_warmup_epochs"]
        if epoch < warmup_epochs:
            lr = self.config["learning_rate"] * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups: 
                param_group['lr'] = lr

    def _run_epoch(self, loader, criterion, is_training=True):
        """运行一个epoch"""
        self.model.train(is_training)
        total_loss, total_acc, num_batches = 0.0, 0.0, 0

        with torch.set_grad_enabled(is_training):
            for batch_idx, (data, target) in enumerate(tqdm(loader, desc=f"{'Train' if is_training else 'Val'}")):
                data, target = data.to(self.device), target.to(self.device)
                
                if is_training:
                    self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                
                if is_training:
                    loss.backward()
                    if self.config.get("max_grad_norm", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    self.optimizer.step()
                    
                    if self.ema is not None:
                        self.ema.update()
                
                total_loss += loss.item()
                total_acc += self._calculate_accuracy(output, target)
                num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def train(self, train_loader, val_loader):
        """训练模型"""
        pos_weights = self._calculate_pos_weights(train_loader.dataset)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        # 检查数据增强配置
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        if not enable_augmentation:
            strategy_name = "no_augmentation"
        elif use_rfboost:
            strategy_name = "rfboost"
        else:
            strategy_name = self.config.get('augment_strategy', 'default')
        
        model_path = os.path.join(self.config["output_dir"], 
                                  f"best_model_{strategy_name}_{self.task}_repeat{self.repeat_idx}.pth")
        early_stopping = EarlyStopping(patience=self.config["early_stopping_patience"], verbose=True)
        
        print(f"训练集大小: {len(train_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
        start_time = time.time()
        
        for epoch in range(self.config["epochs"]):
            self._adjust_learning_rate(epoch)
            
            train_loss, train_acc = self._run_epoch(train_loader, criterion, is_training=True)
            
            if epoch % self.config["validation_frequency"] == 0:
                val_loss, val_acc = self._run_epoch(val_loader, criterion, is_training=False)
                
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                early_stopping(val_loss, self.model, model_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            if epoch >= self.config["learning_rate_warmup_epochs"]:
                self.main_scheduler.step()

        # 加载最佳模型
        self.model.load_state_dict(torch.load(model_path))
        if self.ema is not None:
            self.ema.apply_shadow()
        _, final_val_acc = self._run_epoch(val_loader, criterion, is_training=False)
        if self.ema is not None:
            self.ema.restore()
            
        print(f"Final best validation accuracy: {final_val_acc:.4f}")
        self.plot_training_curves()
        print(f"Training complete in {time.time() - start_time:.2f}s")
        return final_val_acc

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        
        # 准确率
        ax1.plot(self.history['train_acc'], label='Training Accuracy', color='#2E86AB', 
                linewidth=2.5, marker='o', markersize=3, alpha=0.8)
        ax1.plot(self.history['val_acc'], label='Validation Accuracy', color='#A23B72', 
                linewidth=2.5, marker='s', markersize=3, alpha=0.8)
        
        ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim(0.5, 1.0)
        ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        # 损失
        ax2.plot(self.history['train_loss'], label='Training Loss', color='#F18F01', 
                linewidth=2.5, marker='^', markersize=3, alpha=0.8)
        ax2.plot(self.history['val_loss'], label='Validation Loss', color='#C73E1D', 
                linewidth=2.5, marker='v', markersize=3, alpha=0.8)
        
        ax2.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=14, fontweight='bold')
        
        all_losses = self.history['train_loss'] + self.history['val_loss']
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        loss_range = max_loss - min_loss
        ax2.set_ylim(max(0, min_loss - loss_range * 0.1), max_loss + loss_range * 0.1)
        
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        max_epochs = len(self.history['train_acc'])
        tick_interval = max(1, max_epochs // 10)
        ax2.set_xticks(range(0, max_epochs + tick_interval, tick_interval))
        
        # 生成标题
        frequency_band = self.config.get('frequency_band', '2.4')
        environment = self.config.get('environment', 'Classroom').capitalize()
        task = self.task.capitalize()
        
        # 检查数据增强配置
        enable_augmentation = self.config.get("enable_data_augmentation", True)
        use_rfboost = self.config.get("use_rfboost_augmentation", False)
        
        if not enable_augmentation:
            augment_strategy = "NO_AUGMENTATION"
            strategy_name = "no_augmentation"
        elif use_rfboost:
            augment_strategy = "RFBOOST"
            strategy_name = "rfboost"
        else:
            augment_strategy = self.config.get('augment_strategy', 'default').upper()
            strategy_name = self.config.get('augment_strategy', 'default')
        
        model_name = self.config.get('model_name', 'Model')
        
        title = f"{model_name}-{frequency_band}G-{environment}-{task} ({augment_strategy})"
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F4FD", edgecolor="#2E86AB", alpha=0.8))
        
        plt.subplots_adjust(hspace=0.15, top=0.90, bottom=0.08, left=0.08, right=0.95)
        
        # 保存图像
        plot_path = os.path.join(self.config["output_dir"], 
                                f"training_curves_{strategy_name}_{self.task}_repeat{self.repeat_idx}.svg")
        plt.savefig(plot_path, format='svg', bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()


# 便捷函数
def create_trainer(model, device, task, repeat_idx, config):
    """创建训练器的便捷函数"""
    return Trainer(model, device, task, repeat_idx, config)


def create_early_stopping(patience=7, verbose=False, delta=0):
    """创建早停机制的便捷函数"""
    return EarlyStopping(patience, verbose, delta)


def create_ema(model, decay=0.999):
    """创建EMA的便捷函数"""
    return EMA(model, decay)