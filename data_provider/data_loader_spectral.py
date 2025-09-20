import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class Dataset_Spectral(Dataset):
    def __init__(self, args, flag='train', show_stats=False, feature_scaler=None, label_scaler=None):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.is_finetune = getattr(args, 'is_finetune', False)

        self.args = args
        self.root_path = args.root_path
        self.show_stats = getattr(args, 'show_stats', False)

        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.transform = getattr(args, 'train_transform', None) if self.flag == 'train' else None
        self.__read_data__()

    def __read_data__(self):
        feature_path = os.path.join(self.root_path, self.flag, self.args.feature_filename)
        label_path = os.path.join(self.root_path, self.flag, self.args.labels_filename)

        if self.args.feature_filename.endswith('.csv'):
            df_feature = pd.read_csv(feature_path, index_col=0)
        elif self.args.feature_filename.endswith('.feather'):
            df_feature = pd.read_feather(feature_path)
            # Feather格式不会像CSV一样自动处理索引，因此需要手动设置
            # 假设索引列的名称是'obsid'，如果不是，则使用第一列作为索引
            if 'obsid' in df_feature.columns:
                df_feature.set_index('obsid', inplace=True)
            else:
                raise ValueError('必须在特征中提供obsid')
        else:
            raise ValueError(f"不支持的特征文件格式: {self.args.feature_filename}。请使用 .csv 或 .feather。")

        df_label = pd.read_csv(label_path, index_col=0)

        if self.is_finetune:
            print("Filtering for finetuning dataset (FeH < -2)")
            df_label = df_label[df_label['FeH'] < -2]

        common_obsids = df_feature.index.intersection(df_label.index)
        
        shuffled_obsids = common_obsids.tolist()
        if self.flag == 'train':
            random.shuffle(shuffled_obsids)

        df_feature = df_feature.loc[shuffled_obsids]
        df_label = df_label.loc[shuffled_obsids][self.args.targets]
        
        if self.flag == 'train' and self.show_stats:
            print("Statistics for original training data (before scaling):")
            self._calculate_and_print_stats(df_feature, "Features", stat_type='feature')
            self._calculate_and_print_stats(df_label, "Labels", stat_type='label')

        self.data_feature = df_feature.values
        data_label_raw = df_label.values
        self.obsids = shuffled_obsids

        if self.label_scaler:
            self.data_label = self.label_scaler.transform(data_label_raw)
        else:
            self.data_label = data_label_raw
            
        if self.flag == 'train':
            #if self.show_stats:
                print("Statistics for scaled training data:")
                self._calculate_and_print_stats(pd.DataFrame(self.data_feature), "Features", stat_type='feature')
                self._calculate_and_print_stats(pd.DataFrame(data_label_raw), "Labels", stat_type='label')
            
            # if self.transform:
            #     self.data_feature = self.transform(self.data_feature)
            #     print(f"Applied training data augmentations: {self.transform.aval_name}")

        print(f"[{self.__class__.__name__}] flag: {self.flag}")
        print(f"feature shape: {self.data_feature.shape}, label shape: {self.data_label.shape}")

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
        seq_x_feature = self.data_feature[index]
        
        # 使用 np.stack 将同一个归一化谱复制为两个通道，以适配双分支模型输入
        #x_combined = np.stack([seq_x_feature, seq_x_feature], axis=-1)

        seq_y = self.data_label[index]
        obsid = self.obsids[index]

        # 转换为Tensor
        x_combined = torch.from_numpy(seq_x_feature).float() # 最终形状: [L, 2]
        seq_y = torch.from_numpy(seq_y).float()
            
        return x_combined, seq_y, obsid

    def __len__(self):
        return len(self.data_label)