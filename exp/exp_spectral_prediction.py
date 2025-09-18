
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from utils.stellar_metrics import calculate_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot

import os
import yaml

import warnings

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

        # 从数据加载器中取一个样本
        sample_batch, _, _ = next(iter(self.train_loader))
        sample_batch = sample_batch.float().to(self.device)

        # 将样本传递给模型构建函数
        self.model = self._build_model(sample_batch=sample_batch)

    def _get_data(self):
        self.train_data, self.train_loader = data_provider(args=self.args,flag='train', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(args=self.args,flag='val', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        #self.test_data, self.test_loader = data_provider(args=self.args,flag='test', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler) if os.path.exists(os.path.join(self.args.root_path, 'test')) else (None, None)



    def _get_finetune_data(self):
        """
        Loads the finetuning dataset by setting a temporary flag in args.
        """
        # Set is_finetune flag in args
        self.args.is_finetune = True
        self.logger.info("Loading finetuning dataset...")
        
        self.finetune_train_data, self.finetune_train_loader = data_provider(args=self.args, flag='train', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.finetune_vali_data, self.finetune_vali_loader = data_provider(args=self.args, flag='val', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        
        test_path = os.path.join(self.args.root_path, 'test')
        self.finetune_test_data, self.finetune_test_loader = data_provider(args=self.args, flag='test', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler) if os.path.exists(test_path) else (None, None)

        # Unset the flag to avoid side effects in other parts of the code
        self.args.is_finetune = False
        self.logger.info("Finetuning dataset loaded.")



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
