import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import yaml
from utils.stellar_metrics import Scaler

def print_stats(data, stats_type='both', name=None):
    """
    统一打印Pandas DataFrame或NumPy数组的统计信息
    
    参数:
        data: 要分析的数据，可以是Pandas DataFrame或NumPy数组
        name: 数据名称（可选），用于标识输出
        stats_type: 统计信息类型（仅对DataFrame有效）
            'columns' - 只显示每列的统计信息
            'overall' - 只显示总体统计信息
            'both' - 同时显示两者（默认）
    """
    # 设置数据名称
    data_name = name if name is not None else "数据"
    
    # 检查数据是否为空
    if isinstance(data, pd.DataFrame):
        if data.empty:
            print(f"{data_name} 是一个空的DataFrame，没有统计信息可显示。")
            return
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            print(f"{data_name} 是一个空的NumPy数组，没有统计信息可显示。")
            return
    else:
        print("错误：输入数据必须是Pandas DataFrame或NumPy数组。")
        return
    
    # 处理DataFrame
    if isinstance(data, pd.DataFrame):
        print(f"\n===== {data_name} (DataFrame) 统计信息 =====")
        
        # 检查参数有效性
        valid_types = ['columns', 'overall', 'both']
        if stats_type not in valid_types:
            print(f"无效的stats_type参数。必须是以下之一: {', '.join(valid_types)}")
            return
        
        # 打印每列的统计数据（如果需要）
        if stats_type in ['columns', 'both']:
            print("\n--- 各列统计信息 ---")
            column_stats = data.describe(include='all').transpose()
            print(column_stats.round(4))
        
        # 打印总体统计信息（如果需要）
        if stats_type in ['overall', 'both']:
            print("\n--- 总体统计信息 ---")
            all_values = data.stack().values
            overall_stats = {
                '总体最大值': all_values.max(),
                '总体最小值': all_values.min(),
                '总体均值': all_values.mean(),
                '总体中位数': np.median(all_values),
                '总体标准差': all_values.std(),
                '总体总和': all_values.sum(),
                '总体数据点数量': len(all_values)
            }
            for key, value in overall_stats.items():
                print(f"{key}: {round(value, 4)}")
    
    # 处理NumPy数组
    elif isinstance(data, np.ndarray):
        print(f"\n===== {data_name} (NumPy数组) 统计信息 =====")
        
        # 打印总体统计信息
        print("\n--- 总体统计信息 ---")
        flat_arr = data.flatten()
        overall_stats = {
            '总体最大值': flat_arr.max(),
            '总体最小值': flat_arr.min(),
            '总体均值': flat_arr.mean(),
            '总体中位数': np.median(flat_arr),
            '总体标准差': flat_arr.std(),
            '总体总和': flat_arr.sum(),
            '总体数据点数量': len(flat_arr)
        }
        for key, value in overall_stats.items():
            print(f"{key}: {round(value, 4)}")
        
        # 如果是二维数组，额外打印每列统计
        # if data.ndim == 2 and data.shape[1] > 1:
        #     print("\n--- 每列统计信息 ---")
        #     num_cols = data.shape[1]
        #     for col in range(num_cols):
        #         column_data = data[:, col]
        #         col_stats = {
        #             '最大值': column_data.max(),
        #             '最小值': column_data.min(),
        #             '均值': column_data.mean(),
        #             '中位数': np.median(column_data),
        #             '标准差': column_data.std()
        #         }
        #         print(f"\n列 {col} 统计:")
        #         for key, value in col_stats.items():
        #             print(f"  {key}: {round(value, 4)}")
    
    print("\n" + "-"*60)

class Dataset_Spectral(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, feature_scaler=None):
        self.args = args
        self.feature_size = args.feature_size
        self.label_size = args.label_size
        
        # init
        self.flag=flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.targets = args.targets
        self.ratio = args.split_ratio
        self.root_path = args.root_path
        
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        
        self.__read_data__()

    def __read_data__(self):
        # Load the datasets
        features_continuum_df = pd.read_csv(os.path.join(self.root_path, self.args.spectra_continuum_path), dtype={'obsid': 'int64'})
        features_normalized_df = pd.read_csv(os.path.join(self.root_path, self.args.spectra_normalized_path), dtype={'obsid': 'int64'})
        labels_df = pd.read_csv(os.path.join(self.root_path, self.args.label_path), dtype={'obsid': 'int64'})
        
        # Find common obsids
        common_obsids = list(set(features_continuum_df['obsid']) & set(features_normalized_df['obsid']) & set(labels_df['obsid']))
        
        # Shuffle obsids for reproducible random split. Use seed 42 if not provided.
        seed = self.args.seed if hasattr(self.args, 'seed') and self.args.seed is not None else 42
        print(f"Shuffling dataset with random seed: {seed}")
        import random
        random.seed(seed)
        random.shuffle(common_obsids)

        # Filter and align dataframes to the (potentially shuffled) common_obsids
        # First, set 'obsid' as the index to allow for efficient lookup and alignment.
        features_continuum_df.set_index('obsid', inplace=True)
        features_normalized_df.set_index('obsid', inplace=True)
        labels_df.set_index('obsid', inplace=True)

        # Now, re-order the dataframes based on the common_obsids list.
        features_continuum_df = features_continuum_df.loc[common_obsids].reset_index()
        features_normalized_df = features_normalized_df.loc[common_obsids].reset_index()
        labels_df = labels_df.loc[common_obsids].reset_index()

        # Get feature and target columns based on feature_size
        feature_cols_continuum = features_continuum_df.columns[1:self.feature_size+1]
        feature_cols_normalized = features_normalized_df.columns[1:self.feature_size+1]

        # Separate feature sets
        data_x_continuum = features_continuum_df[feature_cols_continuum].values
        data_x_normalized = features_normalized_df[feature_cols_normalized].values

        # 根据您的要求，对归一化谱数据进行裁剪，将数值范围强制限制在[-1, 3]
        print("Applying clipping to normalized spectra data (range: [-1, 3])...")
        data_x_normalized = np.clip(data_x_normalized, -1, 3)

        data_y = labels_df[self.targets].values
        if self.flag =='train': 
            print("连续谱数据来源：",self.root_path, self.args.spectra_continuum_path)
            print("归一化数据来源：",self.root_path, self.args.spectra_normalized_path)
            print('训练集统计信息')
            # 打印features_continuum_df的统计数据
            print_stats(features_continuum_df[feature_cols_continuum],'overall','连续谱数据')
            # 打印features_continuum_df的统计数据
            print_stats(features_normalized_df[feature_cols_normalized],'overall','归一化谱数据')
            print_stats( labels_df[self.targets],'columns','标签数据统计')
        
        # Split data based on ratio
        total_samples = len(common_obsids)
        train_ratio, val_ratio, _ = self.ratio
        train_boundary = int(total_samples * train_ratio)
        val_boundary = int(total_samples * (train_ratio + val_ratio))

        # Transform only the continuum features
        if self.feature_scaler:
            data_x_continuum_scaled = self.feature_scaler.transform(data_x_continuum)
        else:
            raise ValueError("没有提供连续谱归一化器")
            data_x_continuum_scaled = data_x_continuum
        
        # Concatenate scaled continuum features with the original normalized features
        data_x = np.concatenate((data_x_continuum_scaled, data_x_normalized), axis=1)

        # Transform labels if a scaler is provided
        if self.label_scaler:
            data_y_scaled = self.label_scaler.transform(data_y)
        else:
            raise ValueError("没有提供标签归一化器")
            data_y_scaled = data_y
        
        # Define borders for train, val, test sets
        if self.set_type == 0:  # train
            border1, border2 = 0, train_boundary
        elif self.set_type == 1:  # val
            border1, border2 = train_boundary, val_boundary
        else:  # test
            border1, border2 = val_boundary, total_samples
        if self.flag =='train': 
            # print("连续谱数据来源：",self.root_path, self.args.spectra_continuum_path)
            # print("归一化数据来源：",self.root_path, self.args.spectra_normalized_path)
            print('训练集归一化后统计信息')
            # 打印features_continuum_df的统计数据
            print_stats(data_x_continuum_scaled,'overall','连续谱数据')
            # 打印features_continuum_df的统计数据
            print_stats(data_x_normalized,'overall','光谱归一化后数据')
            print_stats( data_y_scaled,'columns','标签数据')
        self.data_x = data_x[border1:border2]
        self.data_y = data_y_scaled[border1:border2]
        self.obsids = labels_df['obsid'].values[border1:border2]
        self.raw_data_y = data_y[border1:border2]

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        obsid = self.obsids[index]
        
        # Dummy values for mark arrays as they are not used in this task
        seq_x_mark = np.zeros((seq_x.shape[0], 1))
        seq_y_mark = np.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, obsid

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        if self.label_scaler:
            return self.label_scaler.inverse_transform(data)
        else:
            return data
            
    def get_raw_targets(self):
        return self.raw_data_y
