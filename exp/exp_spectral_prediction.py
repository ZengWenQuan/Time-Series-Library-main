
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.spectral_prediction import DualBranchMoENet

from utils.tools import EarlyStopping
from utils.stellar_metrics import calculate_metrics, format_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, save_history_plot
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

from utils.stellar_metrics import Scaler

warnings.filterwarnings('ignore')


class Exp_Spectral_Prediction(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """
    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)
        self._setup_logger()
        self.targets = args.targets
        self.args=args
        if hasattr(args, 'model_conf') and args.model_conf and os.path.exists(args.model_conf):
            try:
                with open(args.model_conf, 'r') as f:
                    model_config = yaml.safe_load(f)
                training_settings = model_config.get('training_settings', {})
                self.loss_function_name = training_settings.get('loss_function', 'RegressionFocalLoss')
                use_amp_from_conf = model_config.get('mixed_precision', False)
                if use_amp_from_conf and not getattr(args, 'use_amp', False):
                    self.use_amp = True
                else:
                    self.use_amp = getattr(args, 'use_amp', False)
            except Exception as e:
                self.loss_function_name = 'RegressionFocalLoss'
                self.use_amp = getattr(args, 'use_amp', False)
        else:
            self.loss_function_name = 'mae'
            self.use_amp = getattr(args, 'use_amp', False)

        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        self.label_scaler=self.get_label_scaler()
        self.feature_scaler=self.get_feature_scaler()
        self._get_data()

    def _get_data(self):
        self.train_data, self.train_loader = data_provider(args=self.args,flag='train', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(args=self.args,flag='val', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.test_data, self.test_loader = data_provider(args=self.args,flag='test', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)

    def _select_criterion(self):
        loss_mapping = {
            'l1': nn.L1Loss, 'mae': nn.L1Loss,
            'l2': nn.MSELoss, 'mse': nn.MSELoss,
            'regressionfocalloss': RegressionFocalLoss,
            'gaussiannllloss': GaussianNLLLoss
        }
        loss_class = loss_mapping.get(self.loss_function_name.lower())
        if loss_class:
            return loss_class()
        else:
            return RegressionFocalLoss()

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'cos':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

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
        
    def vali(self, vali_data, vali_loader, criterion):
        if not vali_data: return None, None, None
        total_loss, all_preds, all_trues = [], [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_continuum,batch_x_normalized, batch_y,batch_obsid) in enumerate(vali_loader):
                outputs = self.model(batch_x_continuum.float().to(self.device), batch_x_normalized.float().to(self.device))
                pred, true = outputs.detach(), batch_y.float().detach()
                loss = criterion(pred.to(self.device), true.to(self.device))
                total_loss.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_trues.append(true.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        if self.label_scaler: 
            all_preds = self.label_scaler.inverse_transform(all_preds)
            all_trues = self.label_scaler.inverse_transform(all_trues)
        return np.average(total_loss), all_preds, all_trues

    # --- ADDED: Reusable metric processing function ---
    def calculate_and_save_all_metrics(self, preds, trues, phase, save_as):
        if preds is None or trues is None: return None
        self.logger.info(f"Calculating and saving {save_as} metrics for {phase} set...")
        save_path = os.path.join(self.args.run_dir, 'metrics', save_as)
        
        reg_metrics = calculate_metrics(preds, trues, self.args.targets)
        save_regression_metrics(reg_metrics, save_path, self.args.targets, phase=phase)
        
        cls_metrics = calculate_feh_classification_metrics(preds, trues, self.args.feh_index)
        save_feh_classification_metrics(cls_metrics, save_path, phase=phase)
        return reg_metrics

    def train(self):
        mlflow.set_experiment(self.args.task_name)
        mlflow.start_run(run_name=f"{self.args.model}_{self.args.model_id}")
        mlflow.log_params(vars(self.args))

        chechpoint_path=os.path.join(self.args.run_dir, 'checkpoints')
        os.makedirs(chechpoint_path, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()
        mlflow.log_param("loss_function", criterion.__class__.__name__)

        history_train_loss, history_vali_loss, history_lr = [], [], []

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            for i, (batch_x_continuum, batch_x_normalized, batch_y, batch_obsid) in enumerate(self.train_loader):
                model_optim.zero_grad()
                outputs = self.model(batch_x_continuum.float().to(self.device), batch_x_normalized.float().to(self.device))
                loss = criterion(outputs, batch_y.float().to(self.device))
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            # --- Evaluation (Structure preserved as requested) ---
            train_loss_avg = np.average(train_loss)
            vali_loss, vali_preds, vali_trues = self.vali(self.vali_data, self.vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(self.test_data, self.test_loader, criterion)
            train_eval_loss, train_preds, train_trues = self.vali(self.train_data, self.train_loader, criterion)

            # --- Metric Processing (Refactored into new function) ---
            # Process and save "latest" metrics
            train_metrics = self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "latest")
            vali_metrics = self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "latest")
            test_metrics = self.calculate_and_save_all_metrics(test_preds, test_trues, "test", "latest")
            
            # Manually combine val+test for combined metrics
            if test_preds is not None:
                combined_preds = np.concatenate([vali_preds, test_preds])
                combined_trues = np.concatenate([vali_trues, test_trues])
                combined_metrics = self.calculate_and_save_all_metrics(combined_preds, combined_trues, "combined_val_test", "latest")

            # --- Logging and History ---
            current_lr = model_optim.param_groups[0]['lr']
            history_train_loss.append(train_loss_avg); history_vali_loss.append(vali_loss); history_lr.append(current_lr)
            log_msg = f"Epoch: {epoch + 1} | Train Loss: {train_loss_avg:.4f} | Vali Loss: {vali_loss:.4f}"
            if test_loss is not None: log_msg += f" | Test Loss: {test_loss:.4f}"
            self.logger.info(log_msg)
            
            # Log to MLflow using returned metrics
            mlflow.log_metric('train_loss_epoch', train_loss_avg, step=epoch)
            mlflow.log_metric('val_loss', vali_loss, step=epoch)
            if test_loss is not None: mlflow.log_metric('test_loss', test_loss, step=epoch)
            if vali_metrics: mlflow.log_metric('val_mae', vali_metrics['mae'], step=epoch)

            # --- Early Stopping & Best Model Logic ---
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                self.logger.info(f"New best model found. Saving best metrics...")
                self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "best")
                self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "best")
                if test_preds is not None:
                    self.calculate_and_save_all_metrics(test_preds, test_trues, "test", "best")
                    self.calculate_and_save_all_metrics(combined_preds, combined_trues, "combined_val_test", "best")

            if early_stopping.early_stop: break
            if scheduler is not None: scheduler.step()

            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir)
        mlflow.log_artifact(self.args.run_dir, artifact_path="results")
        mlflow.end_run()
        return self.model
