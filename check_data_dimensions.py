import numpy as np
import os

# 检查一个CSI数据文件的维度
data_dir = "d:/InceptionTime-Attention/pythonProject1/dataset/wifi_csi/amp"
sample_file = "act_1_1.npy"
file_path = os.path.join(data_dir, sample_file)

if os.path.exists(file_path):
    csi_data = np.load(file_path)
    print(f"CSI数据维度: {csi_data.shape}")
    print(f"数据类型: {csi_data.dtype}")
    
    # 如果是4维数据，通常格式为 (time_steps, antennas, subcarriers, ...)
    if len(csi_data.shape) == 4:
        time_steps, antennas, subcarriers, other = csi_data.shape
        print(f"时间步数: {time_steps}")
        print(f"天线数量: {antennas}")
        print(f"子载波数量: {subcarriers}")
        print(f"其他维度: {other}")
        
        # 扁平化后的子载波总数
        total_subcarriers = antennas * subcarriers * other
        print(f"扁平化后总子载波数: {total_subcarriers}")
    
    elif len(csi_data.shape) == 3:
        time_steps, antennas, subcarriers = csi_data.shape
        print(f"时间步数: {time_steps}")
        print(f"天线数量: {antennas}")
        print(f"子载波数量: {subcarriers}")
        
        # 扁平化后的子载波总数
        total_subcarriers = antennas * subcarriers
        print(f"扁平化后总子载波数: {total_subcarriers}")
        
    elif len(csi_data.shape) == 2:
        time_steps, subcarriers = csi_data.shape
        print(f"时间步数: {time_steps}")
        print(f"子载波数量: {subcarriers}")
else:
    print(f"文件不存在: {file_path}")