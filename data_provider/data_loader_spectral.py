import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class Dataset_Spectral(Dataset):
    def __init__(self, args, flag='train', show_stats=False, feature_scaler=None, label_scaler=None, has_labels=True):
        # In prediction mode, flag can be ignored, but for consistency we check it.
        if has_labels:
            assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.has_labels = has_labels
        self.is_finetune = getattr(args, 'is_finetune', False)

        self.args = args
        self.root_path = args.root_path
        self.show_stats = getattr(args, 'show_stats', False)

        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.transform = getattr(args, 'train_transform', None) if self.flag == 'train' else None
        self.__read_data__()

    def __read_data__(self):
        if self.has_labels:
            feature_path = os.path.join(self.root_path, self.flag, self.args.feature_filename)
            label_path = os.path.join(self.root_path, self.flag, self.args.labels_filename)
        else:
            # In prediction mode, root_path is the full path to the feature file.
            feature_path = self.root_path
            label_path = None

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        if feature_path.endswith('.csv'):
            df_feature = pd.read_csv(feature_path, index_col=0)
        elif feature_path.endswith('.feather'):
            df_feature = pd.read_feather(feature_path)
            if 'obsid' in df_feature.columns:
                df_feature.set_index('obsid', inplace=True)
            else:
                raise ValueError('必须在特征中提供obsid')
        else:
            raise ValueError(f"Unsupported feature file format: {self.args.feature_filename}. Use .csv or .feather.")

        if self.has_labels and label_path and os.path.exists(label_path):
            df_label = pd.read_csv(label_path, index_col=0)

            if self.is_finetune:
                print("Filtering for finetuning dataset (FeH < -2)")
                df_label = df_label[df_label['FeH'] < -2 and df_feature['logg']<3.5]

            common_obsids = df_feature.index.intersection(df_label.index)
            
            shuffled_obsids = common_obsids.tolist()
            if self.flag == 'train':
                random.shuffle(shuffled_obsids)

            df_feature = df_feature.loc[shuffled_obsids]
            df_label = df_label.loc[shuffled_obsids][self.args.targets]
            
            data_label_raw = df_label.values
            if self.label_scaler:
                self.data_label = self.label_scaler.transform(data_label_raw)
            else:
                self.data_label = data_label_raw
        else:
            # Create dummy labels if no labels are to be loaded
            num_targets = len(self.args.targets)
            self.data_label = np.zeros((len(df_feature), num_targets))

        self.data_feature = df_feature.values
        self.obsids = df_feature.index.tolist()

        if self.flag == 'train' and self.show_stats and self.has_labels:
            print("Statistics for original training data (before scaling):")
            self._calculate_and_print_stats(df_feature, "Features", stat_type='feature')
            self._calculate_and_print_stats(df_label, "Labels", stat_type='label')

        if self.feature_scaler:
            self.data_feature = self.feature_scaler.transform(self.data_feature)

        print(f"[{self.__class__.__name__}] flag: {self.flag if self.has_labels else 'predict'}")
        print(f"feature shape: {self.data_feature.shape}, label shape: {self.data_label.shape}")

    def _calculate_and_print_stats(self, df, name, stat_type='label'):
        """
        计算并打印DataFrame数据集的统计信息.
        - stat_type='feature': 将所有列视为一个整体，计算总体统计数据.
        - stat_type='label': 分别计算每一列的统计数据.
        """
        print(f"--- Statistics for {name} (type: {stat_type}) ---")
        
        if stat_type == 'feature':
            all_values = pd.Series(df.values.flatten())
            stats_summary = all_values.describe()
        else: # 'label' or default
            stats_summary = df.describe()
            
        print(stats_summary)
        print("---------------------------------")

    def __getitem__(self, index):
        seq_x_feature = self.data_feature[index]
        seq_y = self.data_label[index]
        obsid = self.obsids[index]

        x_combined = torch.from_numpy(seq_x_feature).float()
        seq_y = torch.from_numpy(seq_y).float()
            
        return x_combined, seq_y, obsid

    def __len__(self):
        return len(self.data_feature)