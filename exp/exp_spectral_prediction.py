
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.spectral_prediction import DualBranchMoENet

from utils.stellar_metrics import calculate_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot
from utils.losses import RegressionFocalLoss ,GaussianNLLLoss
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import yaml
from torch.cuda.amp import GradScaler, autocast

import warnings
import pandas as pd

import mlflow

from utils.scaler import Scaler

warnings.filterwarnings('ignore')


class Exp_Spectral_Prediction(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """
    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)
        self.label_scaler=self.get_label_scaler()
        self.feature_scaler=self.get_feature_scaler()
        self._get_data()

    def _get_data(self):
        self.train_data, self.train_loader = data_provider(args=self.args,flag='train', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(args=self.args,flag='val', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.test_data, self.test_loader = data_provider(args=self.args,flag='test', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)



    def get_feature_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f: stats = yaml.safe_load(f)
            return Scaler(scaler_type=self.args.features_scaler_type, stats_dict={'flux': stats['flux']}, target_names=['flux'])
        raise ValueError("没有提供特征统计数据文件路径")

    def get_label_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f: stats = yaml.safe_load(f)
            return Scaler(scaler_type=self.args.label_scaler_type, stats_dict=stats, target_names=self.targets)
        raise ValueError("没有提供标签统计数据文件路径")
        
    # --- ADDED: Reusable metric processing function ---
    def calculate_and_save_all_metrics(self, preds, trues, phase, save_as):
        if preds is None or trues is None: return None
        #self.logger.info(f"Calculating and saving {save_as} metrics for {phase} set...")
        save_path = os.path.join(self.args.run_dir, 'metrics', save_as)
        
        reg_metrics = calculate_metrics(preds, trues, self.args.targets)
        save_regression_metrics(reg_metrics, save_path, self.args.targets, phase=phase)
        
        cls_metrics = calculate_feh_classification_metrics(preds, trues, self.args.feh_index)
        save_feh_classification_metrics(cls_metrics, save_path, phase=phase)
        return reg_metrics
