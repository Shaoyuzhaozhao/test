"""
统一主入口文件
支持单个模型运行和批量模型对比
"""
import argparse
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

import torch
# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from common import Config, set_seed, ExperimentRunner
from model_selector import ModelSelector, MODEL_REGISTRY

# 创建默认配置实例
config_manager = Config()
CONFIG = config_manager.to_dict()  # 使用to_dict()方法获取配置

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='WiFi感知模型训练和评估')
    
    # 模型选择参数
    parser.add_argument('--model', type=str, choices=list(MODEL_REGISTRY.keys()) + ['all'],
                       default='all', help='选择要运行的模型 (默认: all)')
    
    # 实验配置参数
    parser.add_argument('--task', type=str, default='activity',
                       help='任务类型 (默认: activity)')
    parser.add_argument('--frequency_band', type=str, default='5',
                       help='频段 (默认: 5)')
    parser.add_argument('--environment', type=str, default='classroom',
                       help='环境 (默认: classroom)')
    parser.add_argument('--augment_strategy', type=str, default='fda_tda_mre',
                       choices=['none', 'fda', 'tda', 'mre', 'fda_tda', 'tda_mre', 'fda_tda_mre'],
                       help='数据增强策略 (默认: fda_tda_mre)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认: 使用配置文件中的值)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小 (默认: 使用配置文件中的值)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率 (默认: 使用配置文件中的值)')
    parser.add_argument('--repeats', type=int, default=None,
                       help='重复次数 (默认: 使用配置文件中的值)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='设备选择 (默认: auto)')
    parser.add_argument('--list_models', action='store_true',
                       help='列出所有可用模型')
    
    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    config = CONFIG.copy()
    
    # 更新基本配置
    config['task'] = args.task
    config['frequency_band'] = args.frequency_band
    config['environment'] = args.environment
    config['augment_strategy'] = args.augment_strategy
    config['seed'] = args.seed
    
    # 更新训练参数
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.repeats is not None:
        config['repeats'] = args.repeats
    
    # 设备配置
    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device
    
    return config

def run_single_model(model_name, config):
    """运行单个模型"""
    print(f"\n{'='*60}")
    print(f"开始训练模型: {MODEL_REGISTRY[model_name]['name']}")
    print(f"{'='*60}")
    
    # 获取模型类而不是模型实例
    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info['class']
    
    # 创建实验运行器 - 注意参数顺序：config, model_class, model_name
    runner = ExperimentRunner(config, model_class, model_name)
    
    # 运行实验
    results = runner.run()
    
    print(f"\n模型 {MODEL_REGISTRY[model_name]['name']} 训练完成!")
    return results

def run_all_models(config):
    """按顺序运行所有模型"""
    print(f"\n{'='*60}")
    print("开始批量模型训练和对比")
    print(f"{'='*60}")
    
    all_results = {}
    
    for model_name in MODEL_REGISTRY.keys():
        try:
            results = run_single_model(model_name, config)
            all_results[model_name] = results
        except Exception as e:
            print(f"模型 {model_name} 训练失败: {str(e)}")
            all_results[model_name] = None
    
    # 生成详细结果表格
    generate_results_table(all_results, config)
    
    # 生成详细结果表格
    generate_results_table(all_results, config)
    
    return all_results

def generate_results_table(all_results, config):
    """生成详细的结果表格（不影响原有保存策略）"""
    print(f"\n{'='*80}")
    print("批量训练结果汇总表格")
    print(f"{'='*80}")
    
    # 准备表格数据
    table_data = []
    
    for model_name, results in all_results.items():
        model_display_name = MODEL_REGISTRY[model_name]['name']
        
        if results is not None and results:
            # 获取任务结果
            task = config['task']
            if task in results:
                task_result = results[task]
                mean_acc = task_result.get('mean', 0.0)
                std_acc = task_result.get('std', 0.0)
                all_accs = task_result.get('all_results', [])
                
                table_data.append({
                    '模型名称': model_display_name,
                    '平均准确率': f"{mean_acc:.4f}",
                    '标准差': f"±{std_acc:.4f}",
                    '最高准确率': f"{max(all_accs):.4f}" if all_accs else "N/A",
                    '最低准确率': f"{min(all_accs):.4f}" if all_accs else "N/A",
                    '重复次数': len(all_accs),
                    '训练状态': '成功'
                })
            else:
                table_data.append({
                    '模型名称': model_display_name,
                    '平均准确率': "N/A",
                    '标准差': "N/A", 
                    '最高准确率': "N/A",
                    '最低准确率': "N/A",
                    '重复次数': 0,
                    '训练状态': '无结果'
                })
        else:
            table_data.append({
                '模型名称': model_display_name,
                '平均准确率': "N/A",
                '标准差': "N/A",
                '最高准确率': "N/A", 
                '最低准确率': "N/A",
                '重复次数': 0,
                '训练状态': '失败'
            })
    
    # 创建DataFrame并显示
    df = pd.DataFrame(table_data)
    
    # 按平均准确率排序（成功的模型）
    successful_models = df[df['训练状态'] == '成功'].copy()
    failed_models = df[df['训练状态'] != '成功'].copy()
    
    if not successful_models.empty:
        successful_models['平均准确率_数值'] = successful_models['平均准确率'].astype(float)
        successful_models = successful_models.sort_values('平均准确率_数值', ascending=False)
        successful_models = successful_models.drop('平均准确率_数值', axis=1)
    
    # 合并结果
    final_df = pd.concat([successful_models, failed_models], ignore_index=True)
    
    # 打印表格
    print(final_df.to_string(index=False, justify='center'))
    
    # 保存批量训练汇总表格（额外保存，不影响原有保存）
    save_batch_summary_table(final_df, config)
    
    # 打印最佳模型信息
    if not successful_models.empty:
        best_model = successful_models.iloc[0]
        print(f"\n🏆 最佳模型: {best_model['模型名称']}")
        print(f"   平均准确率: {best_model['平均准确率']}")
        print(f"   标准差: {best_model['标准差']}")
        print(f"   最高准确率: {best_model['最高准确率']}")

def save_batch_summary_table(df, config):
    """保存批量训练汇总表格（额外保存，不影响原有的单模型保存策略）"""
    # 创建批量结果汇总目录
    summary_dir = "batch_training_summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    # 生成汇总文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = config['task']
    environment = config['environment']
    frequency_band = config['frequency_band']
    augment_strategy = config['augment_strategy']
    use_rfboost = "rfboost" if config.get('use_rfboost_augmentation', False) else "original"
    
    filename = f"batch_summary_{task}_{environment}_{frequency_band}G_{augment_strategy}_{use_rfboost}_{timestamp}.csv"
    filepath = os.path.join(summary_dir, filename)
    
    # 添加实验配置信息
    config_info = pd.DataFrame([
        ['实验时间', timestamp],
        ['任务类型', task],
        ['环境', environment], 
        ['频段', f"{frequency_band}G"],
        ['增强策略', augment_strategy],
        ['增强器类型', use_rfboost],
        ['训练轮数', config.get('epochs', 'N/A')],
        ['批大小', config.get('batch_size', 'N/A')],
        ['学习率', config.get('learning_rate', 'N/A')],
        ['重复次数', config.get('repeats', 'N/A')],
        ['设备', config.get('device', 'N/A')]
    ], columns=['配置项', '值'])
    
    # 保存到CSV
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        # 写入配置信息
        f.write("批量训练实验配置\n")
        config_info.to_csv(f, index=False)
        f.write("\n模型性能汇总对比\n")
        # 写入结果表格
        df.to_csv(f, index=False)
    
    print(f"\n📊 批量训练汇总表格已保存至: {filepath}")
    print(f"💡 注意: 各模型的详细训练结果仍保存在各自的输出目录中")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 列出模型信息
    if args.list_models:
        selector = ModelSelector()
        selector.list_available_models()
        return
    
    # 更新配置
    config = update_config(args)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    print("WiFi感知模型训练系统")
    print(f"任务: {config['task']}")
    print(f"频段: {config['frequency_band']}")
    print(f"环境: {config['environment']}")
    print(f"增强策略: {config['augment_strategy']}")
    print(f"设备: {config['device']}")
    
    # 运行模型
    if args.model == 'all':
        results = run_all_models(config)
    else:
        results = run_single_model(args.model, config)
    
    print("\n训练完成!")

if __name__ == "__main__":
    main()