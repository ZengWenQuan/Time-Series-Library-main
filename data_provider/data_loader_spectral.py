import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class Dataset_Spectral(Dataset):
    def __init__(self, args, flag='train', show_stats=False, feature_scaler=None, label_scaler=None):
        assert flag in ['train', 'test', 'val']
        self.flag=flag

        self.args = args
        self.root_path = args.root_path
        self.show_stats = getattr(args, 'show_stats', False)

        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler

        self.__read_data__()

    def __read_data__(self):
        continuum_path = os.path.join(self.root_path,self.flag ,self.args.continuum_filename)
        normalized_path = os.path.join(self.root_path,self.flag, self.args.normalized_filename)
        label_path = os.path.join(self.root_path, self.flag,self.args.labels_filename)

        df_continuum = pd.read_csv(continuum_path, index_col=0)
        df_normalized = pd.read_csv(normalized_path, index_col=0)
        df_label = pd.read_csv(label_path, index_col=0)

        common_obsids = df_continuum.index.intersection(df_normalized.index).intersection(df_label.index)
        
        df_continuum_aligned = df_continuum.loc[common_obsids]
        df_normalized_aligned = df_normalized.loc[common_obsids]
        df_label_aligned = df_label.loc[common_obsids]

        if self.flag == 'train' and self.show_stats:
            print("Calculating statistics for original training data...")
            self._calculate_and_print_stats(df_continuum_aligned, "Continuum Spectra", stat_type='feature')
            self._calculate_and_print_stats(df_normalized_aligned, "Normalized Spectra", stat_type='feature')
            self._calculate_and_print_stats(df_label_aligned[self.args.targets], "Labels", stat_type='label')

        shuffled_obsids = common_obsids.tolist()
        random.shuffle(shuffled_obsids)

        df_continuum = df_continuum_aligned.loc[shuffled_obsids]
        df_normalized = df_normalized_aligned.loc[shuffled_obsids]
        df_label = df_label_aligned.loc[shuffled_obsids][self.args.targets]

        data_continuum_raw = df_continuum.values
        self.data_normalized = df_normalized.values
        data_label_raw = df_label.values
        self.obsids = shuffled_obsids

        if self.feature_scaler:
            self.data_continuum = self.feature_scaler.transform(data_continuum_raw)
        else:
            self.data_continuum = data_continuum_raw

        if self.label_scaler:
            self.data_label = self.label_scaler.transform(data_label_raw)
        else:
            self.data_label = data_label_raw
        
        print(f"[{self.__class__.__name__}] flag: {self.flag}")
        print(f"continuum shape: {self.data_continuum.shape}, normalized shape: {self.data_normalized.shape}, label shape: {self.data_label.shape}")

    def _calculate_and_print_stats(self, df, name, stat_type='label'):
        """
        计算并打印DataFrame数据集的统计信息.
        - stat_type='feature': 将所有列视为一个整体，计算总体统计数据.
        - stat_type='label': 分别计算每一列的统计数据.
        """
        print(f"--- Statistics for {name} (type: {stat_type}) ---")
        
        if stat_type == 'feature':
            # 将整个DataFrame展平为一个Series，然后计算总体统计信息
            all_values = pd.Series(df.values.flatten())
            stats_summary = all_values.describe()
        else: # 'label' or default
            # 使用pandas的describe()方法为每一列生成统计摘要
            stats_summary = df.describe()
            
        print(stats_summary)
        print("---------------------------------")

    def __getitem__(self, index):
        seq_x_continuum = self.data_continuum[index]
        seq_x_normalized = self.data_normalized[index]
        seq_y = self.data_label[index]
        obsid = self.obsids[index]

        seq_x_continuum = torch.from_numpy(seq_x_continuum).float()
        seq_x_normalized = torch.from_numpy(seq_x_normalized).float()
        seq_y = torch.from_numpy(seq_y).float()

        if seq_x_continuum.ndim == 1:
            seq_x_continuum = seq_x_continuum.unsqueeze(-1)
        if seq_x_normalized.ndim == 1:
            seq_x_normalized = seq_x_normalized.unsqueeze(-1)
            
        return seq_x_continuum, seq_x_normalized, seq_y, obsid

    def __len__(self):
        return len(self.data_label)

    def inverse_transform_label(self, data):
        if self.label_scaler:
            return self.label_scaler.inverse_transform(data)
        return data