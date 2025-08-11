import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class Dataset_Steller(Dataset):
    def __init__(self, args, flag='train', feature_scaler=None, label_scaler=None):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.args = args
        if feature_scaler == None or label_scaler == None:
            raise ValueError('scaler error!!!!')
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.show_stats = getattr(args, 'show_stats', False)
        self.__read_data__()

    def _calculate_and_print_stats(self, df, name, stat_type='label'):
        """
        计算并打印DataFrame数据集的统计信息.
        - stat_type='feature': 将所有列视为一个整体，计算总体统计数据.
        - stat_type='label': 分别计算每一列的统计数据.
        """
        print(f"--- Statistics for {name} (type: {stat_type}) ---")
        if df.empty:
            print("DataFrame is empty. No stats to calculate.")
            print("---------------------------------")
            return
            
        if stat_type == 'feature':
            all_values = pd.Series(df.values.flatten())
            stats_summary = all_values.describe()
        else: # 'label' or default
            stats_summary = df.describe()
            
        print(stats_summary)
        print("---------------------------------")

    def __read_data__(self):
        # 1. 构建路径
        split_path = os.path.join(self.args.root_path, self.flag)
        feature_path = os.path.join(split_path, self.args.feature_filename)
        label_path = os.path.join(split_path, self.args.labels_filename)

        # 2. 加载数据
        df_features = pd.read_csv(feature_path, index_col=0)
        df_labels = pd.read_csv(label_path, index_col=0)

        # 3. 对齐和打乱
        common_obsids = df_features.index.intersection(df_labels.index)
        df_features = df_features.loc[common_obsids]
        df_labels = df_labels.loc[common_obsids]
        self.shuffled_obsids = common_obsids.tolist()
        random.shuffle(self.shuffled_obsids)
        df_features = df_features.loc[self.shuffled_obsids]
        df_labels = df_labels.loc[self.shuffled_obsids]

        # 4. 提取Numpy数组
        features_raw = df_features.values
        labels_raw = df_labels[self.args.targets].values

        # 5. 应用Scaler
        features_scaled = self.feature_scaler.transform(features_raw)
        labels_scaled = self.label_scaler.transform(labels_raw)

        # --- 新增：在训练集上打印归一化前后的统计数据对比 ---
        if self.flag == 'train' and self.show_stats:
            print("\n================== Data Statistics Comparison (Train Set) ==================")
            # 特征统计
            self._calculate_and_print_stats(df_features, "Features (Before Scaling)", stat_type='feature')
            self._calculate_and_print_stats(pd.DataFrame(features_scaled), "Features (After Scaling)", stat_type='feature')
            # 标签统计
            self._calculate_and_print_stats(df_labels[self.args.targets], "Labels (Before Scaling)", stat_type='label')
            self._calculate_and_print_stats(pd.DataFrame(labels_scaled, columns=self.args.targets), "Labels (After Scaling)", stat_type='label')
            print("============================================================================\n")

        self.data_x = features_scaled
        self.data_y = labels_scaled

    def __getitem__(self, index):
        # 增加返回 obsid，与其他数据加载器保持一致
        return torch.from_numpy(self.data_x[index]).float(), torch.from_numpy(self.data_y[index]).float(), self.shuffled_obsids[index]

    def __len__(self):
        return len(self.data_x)