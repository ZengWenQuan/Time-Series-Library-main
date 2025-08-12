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
        # --- 从args中获取预先构建好的transform流水线 ---
        self.transform = getattr(args, 'train_transform', None) if self.flag == 'train' else  None
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
        self.shuffled_obsids = common_obsids.tolist()
        if self.flag=='train': random.shuffle(self.shuffled_obsids)
        df_features = df_features.loc[self.shuffled_obsids]
        df_labels = df_labels.loc[self.shuffled_obsids]

        # 4. 提取Numpy数组
        features_raw = df_features.values
        labels_raw = df_labels[self.args.targets].values

        # 5. 应用Scaler
        features_scaled = self.feature_scaler.transform(features_raw)
        labels_scaled = self.label_scaler.transform(labels_raw)

        # --- 新增：在训练集上打印归一化前后的统计数据对比 ---
        if self.flag == 'train':
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

    def _build_transforms(self):
        """Helper function to build augmentation pipeline from config."""
        from utils.augmentations import AUGMENTATION_REGISTRY, Compose, ProbabilisticAugmentation
        import yaml
        augs_list = []
        if hasattr(self.args, 'stats_path') and self.args.stats_path and os.path.exists(self.args.stats_path):
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            augs_conf = stats.get('augs_conf', [])
            for aug_conf in augs_conf:
                if aug_conf.get('enabled', False) and aug_conf['name'] in AUGMENTATION_REGISTRY:
                    AugmentationClass = AUGMENTATION_REGISTRY[aug_conf['name']]
                    transform = AugmentationClass(**aug_conf.get('params', {}))
                    prob_transform = ProbabilisticAugmentation(transform, p=aug_conf.get('p', 1.0))
                    augs_list.append(prob_transform)
        return Compose(augs_list) if augs_list else None

    def __getitem__(self, index):
        x = self.data_x[index].copy() # 使用copy避免修改原始数据
        y = self.data_y[index]
        obsid = self.shuffled_obsids[index]

        # 如果是训练集且transform存在，则应用数据增强
        if self.transform:
            x = self.transform(x)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), obsid

    def __len__(self):
        return len(self.data_x)