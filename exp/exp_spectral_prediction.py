
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from utils.stellar_metrics import calculate_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot

import os
import yaml

import warnings

from utils.scaler import Scaler
import pandas as pd

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

    def test_all(self):
        # 1. Load model
        checkpoint_path = self.args.checkpoints
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            self.logger.error(f"No valid checkpoint path provided via --checkpoints. Aborting test_all.")
            return
        self.logger.info(f"Loading model from provided checkpoint: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)

        # 2. Get data path and create results path
        test_data_root = self.args.test_data_path
        if not test_data_root or not os.path.isdir(test_data_root):
            self.logger.error(f"Invalid test_data_path: {test_data_root}")
            return
            
        results_root = os.path.join(os.path.dirname(test_data_root.rstrip('/')), 'test_all_results_' + os.path.basename(test_data_root.rstrip('/')))
        self.logger.info(f"Results will be saved in: {results_root}")
        os.makedirs(results_root, exist_ok=True)

        # 3. Find all subdirectories (each is a dataset)
        dataset_flags = [d for d in os.listdir(test_data_root) if os.path.isdir(os.path.join(test_data_root, d))]

        if not dataset_flags:
            self.logger.warning(f"No subdirectories found in {test_data_root}")
            return

        original_root_path = self.args.root_path
        self.args.root_path = test_data_root

        for flag in dataset_flags:
            self.logger.info(f"--- Starting evaluation on '{flag}' data ---")
            
            try:
                data, loader = data_provider(args=self.args, flag=flag, feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
            except FileNotFoundError as e:
                self.logger.error(f"Could not load data for '{flag}'. Missing file: {e}. Skipping.")
                continue

            if data is None or loader is None:
                self.logger.warning(f"No data/loader for '{flag}' split. Skipping.")
                continue

            save_dir = os.path.join(results_root, flag)
            os.makedirs(save_dir, exist_ok=True)

            loss, preds, trues, obsids = self.vali(data, loader, self._select_criterion())

            if preds is None:
                self.logger.warning(f"Evaluation on '{flag}' returned no results. Skipping.")
                continue

            self.logger.info(f"Saving predictions for '{flag}' to {save_dir}")
            pred_df = pd.DataFrame({'obsid': obsids})
            for i, target_name in enumerate(self.targets):
                pred_df[f'{target_name}_true'] = trues[:, i]
                pred_df[f'{target_name}_pred'] = preds[:, i]
            pred_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

            self.logger.info(f"Calculating and saving metrics for '{flag}'...")
            original_run_dir = self.args.run_dir
            self.args.run_dir = save_dir
            
            self.calculate_and_save_all_metrics(preds, trues, phase=flag, save_as='final')
            
            self.args.run_dir = original_run_dir

        self.args.root_path = original_root_path
        self.logger.info("--- test_all completed ---")
