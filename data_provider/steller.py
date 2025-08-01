import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset



from utils.stellar_metrics import Scaler
class Dataset_Stellar(Dataset):
    def __init__(self, args,flag):
        # size [seq_len, label_len, label_size]
        self.args = args
        
        # info
        self.seq_len = args.feature_size  # 光谱数据长度
        self.label_size = args.label_size  # 需要预测的标签数量
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        #self.features =features
        self.targets = self.args.targets
        self.ratio=self.args.split_ratio
            
        self.features_scaler = Scaler(scaler_type=self.args.features_scaler_type) 

        self.label_scaler = Scaler(scaler_type=self.args.label_scaler_type)
        
        self.root_path = self.args.root_path
        self.data_path = self.args.data_path
        
        
        # 添加标签缩放类型
        
        self.__read_data__()

    def __read_data__(self):
        
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        # 确定训练、验证、测试集的划分比例
        total_samples = len(df_raw)
        train_ratio, val_ratio, test_ratio = self.ratio
        train_boundary = int(total_samples * train_ratio) # 训练集边界
        val_boundary = int(total_samples * (train_ratio + val_ratio)) # 验证集边界
        
        # 根据set_type划分数据集
        if self.set_type == 0:   # train
            border1, border2 = 0, train_boundary 
        elif self.set_type == 1: # val
            border1, border2 = train_boundary, val_boundary
        else:                    # test
            border1, border2 = val_boundary, total_samples
        
        # 分离特征和标签
        cols_data = df_raw.columns[1:self.seq_len+1]  # 跳过obsid列，选择所有光谱数据列
        df_data = df_raw[cols_data]
        
        # 数据标准化处理
        if self.features_scaler:
            train_data = df_data[:train_boundary] # 训练集数据
            self.features_scaler.fit(train_data.values) # 拟合训练集数据
            self.features = self.features_scaler.transform(df_data.values) # 标准化数据
        else:
            self.features = df_data.values
            
        
        # 根据实际需要预测的标签选择标签列
        self.targets = df_raw[self.targets].values
        
        # 保存原始标签数据（未缩放）用于反归一化
        self.raw_targets = self.targets.copy()
        
        
        # 仅在训练数据上拟合
        self.label_scaler.fit(self.targets[:train_boundary]) # 拟合训练集数据
        
        # 对所有数据进行变换
        self.targets = self.label_scaler.transform(self.targets)
        
        # 选择当前分区的数据
        self.data_x = self.features[border1:border2]
        self.data_y = self.targets[border1:border2]
        
        # 保存当前分区的原始标签
        self.raw_targets = self.raw_targets[border1:border2]
        
        # 如果是训练集且启用了FeH采样
        if self.set_type == 0 and hasattr(self.args, 'use_feh_sampling') and self.args.use_feh_sampling:
            try:
                from utils.stellar_sampling import apply_stellar_sampling
                
                # 确定FeH在标签中的索引位置
                feh_index = 2  # 默认FeH是第三个标签 (Teff, logg, FeH, CFe)
                if hasattr(self.args, 'feh_index'):
                    feh_index = self.args.feh_index
                
                # 应用过采样/欠采样
                sampling_strategy = 'balanced'
                if hasattr(self.args, 'feh_sampling_strategy'):
                    sampling_strategy = self.args.feh_sampling_strategy
                
                k_neighbors = 5
                if hasattr(self.args, 'feh_sampling_k_neighbors'):
                    k_neighbors = self.args.feh_sampling_k_neighbors
                
                print(f"应用基于FeH的过采样/欠采样，策略: {sampling_strategy}, k邻居数: {k_neighbors}")
                self.data_x, self.data_y = apply_stellar_sampling(
                    self.data_x, 
                    self.data_y,
                    sampling_strategy=sampling_strategy,
                    random_state=self.args.seed if hasattr(self.args, 'seed') else 42,
                    k_neighbors=k_neighbors,
                    feh_index=feh_index
                )
                
                print(f"重采样后数据形状: X={self.data_x.shape}, y={self.data_y.shape}")
                
                # 重新计算原始标签（用于反归一化）
                self.raw_targets = self.label_scaler.inverse_transform(self.data_y)
            except Exception as e:
                print(f"应用FeH采样时出错: {str(e)}")
                print("继续使用原始数据...")

    def __getitem__(self, index):
        # 因为是回归任务，我们只需要光谱数据和对应的标签
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        
            # 原始数据维度
        seq_x = seq_x.reshape(self.seq_len)
            
        seq_y = seq_y.reshape(self.label_size)
        
        # 占位符，不实际使用
        seq_x_mark = np.zeros((self.seq_len, 1))
        seq_y_mark = np.zeros((self.label_size, 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)
            
    def get_raw_targets(self):
        """
        获取原始的未缩放标签数据
        """
        return self.raw_targets
