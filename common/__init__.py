# common模块初始化文件
# 用于导入和管理通用功能模块

# 配置管理模块
from .config import Config

# 数据增强模块
from .augmentation import (
    OnTheFlyAugmenter,
    AugmentationStrategy,
    create_augmenter,
    create_augmentation_strategy
)

# RFBoost增强模块
from .r_augmentation import (
    RFBoostAugmenter,
    RFBoostAugmentationStrategy,
    create_rfboost_augmenter,
    create_rfboost_augmentation_strategy
)

# 数据集模块
from .dataset import (
    WiMANDataset,
    create_dataset,
    load_annotation_data,
    filter_annotation_data
)

# 训练工具模块
from .training import (
    EMA,
    EarlyStopping,
    Trainer,
    create_trainer,
    create_early_stopping,
    create_ema
)

# 实验管理模块
from .experiment import (
    ExperimentRunner,
    ResultsManager,
    create_experiment_runner,
    create_results_manager,
    run_experiment
)

# 工具函数模块
from .utils import (
    set_seed,
    get_device,
    count_parameters,
    format_time,
    save_json,
    load_json,
    ensure_dir,
    get_memory_usage,
    clear_memory,
    validate_config,
    setup_logging,
    calculate_accuracy,
    normalize_tensor,
    print_model_summary,
    Timer,
    create_timer,
    print_separator
)

# 版本信息
__version__ = "1.0.0"

# 模块列表
__all__ = [
    # 配置管理
    "Config",
    
    # 数据增强
    "OnTheFlyAugmenter",
    "AugmentationStrategy", 
    "create_augmenter",
    "create_augmentation_strategy",
    
    # RFBoost数据增强
    "RFBoostAugmenter",
    "RFBoostAugmentationStrategy",
    "create_rfboost_augmenter", 
    "create_rfboost_augmentation_strategy",
    
    # 数据集
    "WiMANDataset",
    "create_dataset",
    "load_annotation_data",
    "filter_annotation_data",
    
    # 训练工具
    "EMA",
    "EarlyStopping",
    "Trainer",
    "create_trainer",
    "create_early_stopping",
    "create_ema",
    
    # 实验管理
    "ExperimentRunner",
    "ResultsManager",
    "create_experiment_runner",
    "create_results_manager",
    "run_experiment",
    
    # 工具函数
    "set_seed",
    "get_device",
    "count_parameters",
    "format_time",
    "save_json",
    "load_json",
    "ensure_dir",
    "get_memory_usage",
    "clear_memory",
    "validate_config",
    "setup_logging",
    "calculate_accuracy",
    "normalize_tensor",
    "print_model_summary",
    "Timer",
    "create_timer",
    "print_separator",
]