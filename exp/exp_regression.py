import os
import yaml
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import time
import mlflow
from torch.utils.data import DataLoader, ConcatDataset

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.scaler import Scaler
from utils.tools import EarlyStopping
from utils.stellar_metrics import calculate_metrics, format_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot

class Exp_Regression(Exp_Basic):
    def __init__(self, args):
        # 首先调用父类的构造函数，它会初始化 self.args, self.device, self.logger, self.model
        super(Exp_Regression, self).__init__(args)
        
        # --- 核心修正：在这里初始化并赋值 Scalers ---
        self.feature_scaler = self.get_feature_scaler()
        self.label_scaler = self.get_label_scaler()
        
        # 然后使用初始化好的 scalers 去加载数据
        self._get_data()

    def _get_data(self):
        self.train_data, self.train_loader = data_provider(self.args, 'train',feature_scaler= self.feature_scaler,label_scaler= self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(self.args, 'val',feature_scaler= self.feature_scaler, label_scaler=self.label_scaler)
        self.test_data, self.test_loader = data_provider(self.args, 'test', feature_scaler=self.feature_scaler,label_scaler= self.label_scaler)

    # --- 清理后的、唯一的 Scaler 加载逻辑 ---
    def get_feature_scaler(self):
        """为输入特征（光谱flux）加载scaler"""
        if self.args.stats_path and os.path.exists(self.args.stats_path):
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            if 'flux' in stats:
                self.logger.info("Found 'flux' key in stats.yaml for feature scaling.")
                # 关键：特征scaler只使用 'flux' 作为target_name
                return Scaler(scaler_type=self.args.features_scaler_type, 
                              stats_dict={'flux': stats['flux']}, 
                              target_names=['flux'])
        self.logger.warning("Feature stats file not found or 'flux' key missing. Feature scaler will not be used.")
        return None

    def get_label_scaler(self):
        """为目标标签（Teff, logg等）加载scaler"""
        if self.args.stats_path and os.path.exists(self.args.stats_path):
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            self.logger.info("Loading label stats from stats.yaml.")
            # 关键：标签scaler使用 self.targets 列表来加载对应的统计数据
            return Scaler(scaler_type=self.args.label_scaler_type, 
                          stats_dict=stats, 
                          target_names=self.args.targets)
        self.logger.warning("Label stats file not found. Label scaler will not be used.")
        return None
