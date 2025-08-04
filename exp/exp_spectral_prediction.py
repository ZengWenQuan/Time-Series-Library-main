from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import *

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.stellar_metrics import calculate_metrics, format_metrics, calculate_feh_classification_metrics, format_feh_classification_metrics
from utils.stellar_metrics import save_regression_metrics, save_feh_classification_metrics
from utils.losses import RegressionFocalLoss

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import yaml
import logging
import random
from torch.cuda.amp import GradScaler, autocast

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlflow
import matplotlib.pyplot as plt

from utils.stellar_metrics import Scaler

warnings.filterwarnings('ignore')

def fix_seed_worker(worker_id):
    """
    为DataLoader的每个worker设置随机种子
    """
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


class Exp_Spectral_Prediction(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """
    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)
        self._setup_logger()
        # 恒星参数名称
        self.targets = args.targets
        self.args=args
        
        # 混合精度训练设置
        self.use_amp = getattr(args, 'use_amp', False)
        if self.use_amp:
            print("✓ 混合精度训练已启用 (AMP)")
            self.scaler = GradScaler()
        else:
            print("✓ 使用标准精度训练")
            self.scaler = None
        
        # 如果有模型配置文件，检查其中的混合精度设置
        if hasattr(args, 'model_conf') and args.model_conf and os.path.exists(args.model_conf):
            try:
                with open(args.model_conf, 'r') as f:
                    model_config = yaml.safe_load(f)
                    
                # 从配置文件中读取混合精度设置
                config_amp = model_config.get('mixed_precision', False)
                if config_amp and not self.use_amp:
                    print("✓ 从模型配置文件中启用混合精度训练")
                    self.use_amp = True
                    self.scaler = GradScaler()
                    
            except Exception as e:
                print(f"警告: 无法从配置文件中读取混合精度设置: {e}")
        
        self.label_scaler=self.get_label_scaler()
        self.feature_scaler=self.get_feature_scaler()
        info_model=self.args.run_dir+'/model.txt'
        with open(info_model,'w') as f:
            # 写入模型结构
            f.write("模型结构:\n")
            f.write(f"{self.model}\n\n")
            
            # 写入每层参数数量
            f.write("每层参数数量:\n")
            sum_param=0
            for name, param in self.model.named_parameters():                
                f.write(f"  {name}: {param.numel():,} 参数\n")
                sum_param+=param.numel()
            f.write(f'总参数量：{sum_param}')
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.label_scaler, self.feature_scaler)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
  
    def _setup_logger(self):
        import datetime
        # Prevent the logger from propagating to the root logger 
        self.logger = logging.getLogger('CEMP_search')        
        self.logger.setLevel(logging.INFO)   
        self.logger.propagate = False
        
        # 获取当前北京时间
        beijing_time = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        time_str = beijing_time.strftime('%Y-%m-%d %H:%M')
        
        # Formatter        
        formatter = logging.Formatter(f'CEMP search - {time_str} - %(message)s')
        
        # File Handler        
        log_file = os.path.join(self.args.run_dir, 'training.log')        
        file_handler = logging.FileHandler(log_file)        
        file_handler.setLevel(logging.INFO)        
        file_handler.setFormatter(formatter)
        
        # Stream Handler (for console output)        
        stream_handler = logging.StreamHandler()        
        stream_handler.setLevel(logging.INFO)        
        stream_handler.setFormatter(formatter)
        
        # Add handlers to the logger        
        if not self.logger.handlers:            
            self.logger.addHandler(file_handler)            
            self.logger.addHandler(stream_handler)    
               

    def get_feature_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            
            return Scaler(scaler_type=self.args.features_scaler_type, stats_dict={'flux': stats['flux']}, target_names=['flux'])
        else:
            df_continuum = pd.read_csv(os.path.join(self.args.root_path, self.args.spectra_continuum_path))
            feature_cols_continuum = df_continuum.columns[1:self.args.feature_size+1]
            data_x_continuum = df_continuum[feature_cols_continuum].values

            total_samples = len(data_x_continuum)
            train_ratio, _, _ = self.args.split_ratio
            train_boundary = int(total_samples * train_ratio)

            feature_scaler = Scaler(scaler_type=self.args.features_scaler_type)
            feature_scaler.fit(data_x_continuum[:train_boundary], ['flux'])
            return feature_scaler

    def get_label_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)

            return Scaler(scaler_type=self.args.label_scaler_type, stats_dict=stats, target_names=self.targets)
        else:
            df_raw = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
            total_samples = len(df_raw)
            train_ratio, val_ratio, test_ratio = self.args.split_ratio
            train_boundary = int(total_samples * train_ratio)
            
            targets = df_raw[self.targets].values
            if self.args.label_scaler_type:
                label_scaler = Scaler(scaler_type=self.args.label_scaler_type)
                label_scaler.fit(targets[:train_boundary], self.targets)
                return label_scaler
            else:
                return None
        
    def vali(self, vali_data, vali_loader, criterion):
        if vali_data is None:
            return None, None, None
        total_loss = []
        self.model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_obsid) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach()
                true = batch_y.detach()
                loss = criterion(pred, true)
                total_loss.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_trues.append(true.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        
        if self.label_scaler:
            all_preds = self.label_scaler.inverse_transform(all_preds)
            all_trues = self.label_scaler.inverse_transform(all_trues)
        #print(all_preds)
        #print(all_trues)
        return np.average(total_loss) , all_preds , all_trues

    def train(self,setting):
        # --- MLflow Setup ---
        # 设置实验名称，如果不存在则会自动创建
        mlflow.set_experiment(self.args.task_name)
        # 开始一次MLflow运行，所有记录都将保存在这个运行下
        mlflow.start_run(run_name=f"{self.args.model}_{self.args.model_id}")
        # 记录所有超参数
        mlflow.log_params(vars(self.args))

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Conditionally load test data only if the test ratio is greater than zero
        if self.args.split_ratio[2] > 0:
            test_data, test_loader = self._get_data(flag='test')
        else:
            self.logger.info("Test ratio is 0, skipping test set loading and evaluation.")
            test_data, test_loader = None, None

        chechpoint_path=self.args.run_dir+'/'+'checkpoints'
        if not os.path.exists(chechpoint_path):
            os.makedirs(chechpoint_path)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        history_train_loss = []
        history_vali_loss = []
        history_lr = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            grad_norm_before = 0
            grad_norm_after = 0
            
            # --- Gradient Clipping & Logging ---
            max_norm = self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 20.0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_obsid) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 混合精度训练
                if self.use_amp and self.scaler is not None:
                    with autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    # Check for abnormally high loss
                    if hasattr(self.args, 'loss_threshold') and loss.item() > self.args.loss_threshold:
                        print(f"[Warning] Batch {i+1}/{train_steps}: Loss {loss.item():.2f} exceeds threshold {self.args.loss_threshold}. Skipping batch.")
                        continue
                    
                    # 缩放loss并反向传播
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪（在scaled gradients上）
                    self.scaler.unscale_(model_optim)
                    grad_norm_before_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                    if torch.isfinite(grad_norm_before_tensor):
                        grad_norm_before = grad_norm_before_tensor.item()
                    else:
                        grad_norm_before = float('inf')
                    
                    # 更新参数
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    # 标准精度训练
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    # Check for abnormally high loss
                    if hasattr(self.args, 'loss_threshold') and loss.item() > self.args.loss_threshold:
                        print(f"[Warning] Batch {i+1}/{train_steps}: Loss {loss.item():.2f} exceeds threshold {self.args.loss_threshold}. Skipping batch.")
                        continue
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    grad_norm_before_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                    if torch.isfinite(grad_norm_before_tensor):
                        grad_norm_before = grad_norm_before_tensor.item()
                    else:
                        grad_norm_before = float('inf')
                    
                    model_optim.step()

                # Calculate norm after clipping for logging
                norm_after_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        norm_after_sq += param_norm.item() ** 2
                grad_norm_after = norm_after_sq ** 0.5
                # --- End Gradient Clipping & Logging ---

                model_optim.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            
            do_validation = (epoch + 1) % self.args.vali_interval == 0
            is_last_epoch = epoch == self.args.train_epochs - 1

            vali_loss, vali_preds, vali_trues = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(test_data, test_loader, criterion)
            
            history_train_loss.append(train_loss)
            history_vali_loss.append(vali_loss)

            # 在调整学习率前记录当前学习率
            current_lr = model_optim.param_groups[0]['lr']
            history_lr.append(current_lr)

            left_time = (self.args.train_epochs - epoch) *(time.time() - epoch_time)
            
            grad_info = f"Grad Norm: {grad_norm_before:.2f}->{grad_norm_after:.2f}"

            log_message = (
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.4f} Vali Loss: {vali_loss:.4f} "
            )
            if test_data is not None:
                log_message += f"Test Loss: {test_loss:.4f} "
            log_message += (
                f"Vali Interval: {self.args.vali_interval},"
                f"epoch_time: {(time.time() - epoch_time):.2f}s  left_time: {left_time//3600:.0f}h{(left_time%3600)//60:.0f}m{left_time%60:.0f}s | "
                f"lr:{current_lr:.2f} "
                f"{grad_info}"
            )
            self.logger.info(log_message)
            
            metrics_dict_vali = calculate_metrics(vali_preds, vali_trues, self.args.targets)
            feh_metrics_vali = calculate_feh_classification_metrics(vali_preds, vali_trues)
            if test_data is not None:
                metrics_dict_test = calculate_metrics(test_preds, test_trues, self.args.targets)
                feh_metrics_test = calculate_feh_classification_metrics(test_preds, test_trues)
                
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                save_dir = self.args.run_dir+"/metrics/best/"
                save_regression_metrics(metrics_dict_vali, save_dir, self.targets, phase="vali")
                save_feh_classification_metrics(feh_metrics_vali, save_dir, phase="vali")
                if test_data is not None:
                    save_regression_metrics(metrics_dict_test, save_dir, self.targets, phase="test")
                    save_feh_classification_metrics(feh_metrics_test, save_dir, phase="test")

            save_dir = self.args.run_dir+"/metrics/last/"
            save_regression_metrics(metrics_dict_vali, save_dir, self.targets, phase="vali")
            save_feh_classification_metrics(feh_metrics_vali, save_dir, phase="vali")
            if test_data is not None:
                save_regression_metrics(metrics_dict_test, save_dir, self.targets, phase="test")
                save_feh_classification_metrics(feh_metrics_test, save_dir, phase="test")
            if do_validation or is_last_epoch :
                print("验证集指标:")
                print(format_metrics(metrics_dict_vali))
                print("\n验证集FeH分类指标:")
                print(format_feh_classification_metrics(feh_metrics_vali))
                if test_data is not None:
                    print("\n测试集指标:")
                    print(format_metrics(metrics_dict_test))
                    print("\n测试集FeH分类指标:")
                    print(format_feh_classification_metrics(feh_metrics_test))
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 使用通用函数绘制损失和学习率曲线
            self._plot_curve({'Train Loss': history_train_loss, 'Validation Loss': history_vali_loss}, 'Loss', 'Loss', 'loss')
            self._plot_curve({'Learning Rate': history_lr}, 'Learning Rate', 'Learning Rate', 'lr')

            # --- MLflow Logging ---
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            if vali_loss is not None:
                mlflow.log_metric('val_loss', vali_loss, step=epoch)
            mlflow.log_metric('learning_rate', current_lr, step=epoch)

            # Log MAE for each validation target, excluding the overall 'mae'
            if metrics_dict_vali:
                for metric_name, metric_value in metrics_dict_vali.items():
                    # Log per-label MAE (e.g., mae_Teff), but skip the overall 'mae'
                    if 'mae' in metric_name.lower() and metric_name.lower() != 'mae':
                        mlflow.log_metric(f'val_{metric_name}', metric_value, step=epoch)
            
            # Log MAE for each test target, excluding the overall 'mae'
            if test_data is not None and metrics_dict_test:
                for metric_name, metric_value in metrics_dict_test.items():
                    # Log per-label MAE (e.g., mae_Teff), but skip the overall 'mae'
                    if 'mae' in metric_name.lower() and metric_name.lower() != 'mae':
                        mlflow.log_metric(f'test_{metric_name}', metric_value, step=epoch)

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        last_model_path = chechpoint_path + '/' + 'last.pth'
        torch.save(self.model.state_dict(), last_model_path)
        print("the train is over,save the best model and the last model")
        
        # --- MLflow Artifacts & Model Registration ---
        self.logger.info("Logging artifacts and registering model to MLflow...")

        # 1. Log plots as artifacts
        mlflow.log_artifact(os.path.join(self.args.run_dir, 'loss_curve.pdf'))
        mlflow.log_artifact(os.path.join(self.args.run_dir, 'lr_curve.pdf'))

        # 2. Register the best model to the MLflow Model Registry
        # Note: EarlyStopping saves the best model as 'best.pth'
        best_model_path = os.path.join(chechpoint_path, 'best.pth')
        if os.path.exists(best_model_path):
            self.logger.info(f"Registering model '{self.args.model}' from {best_model_path}")
            # Load the best model's weights before registration
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            
            # Use mlflow.pytorch.log_model for registration
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="model",  # Subdirectory within the run's artifacts
                registered_model_name=self.args.model  # This is the key for registration
            )
            self.logger.info(f"Model '{self.args.model}' registered successfully.")
        else:
            self.logger.warning(f"Could not find best model at '{best_model_path}' to register.")

        # 3. End the MLflow run
        mlflow.end_run()

        return self.model

    def _plot_curve(self, data_history, title, y_label, file_suffix):
        """通用绘图函数，可绘制损失或学习率等曲线"""
        plt.figure(figsize=(10, 6))
        for label, values in data_history.items():
            plt.plot(values, label=label)
        
        plt.title(f'Training and Validation {title}')
        plt.xlabel('Epoch')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.args.run_dir, f'{file_suffix}_curve.pdf')
        plt.savefig(save_path, format='pdf')
        plt.close()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.run_dir, 'checkpoints', 'checkpoint.pth')))

        preds = []
        trues = []
        obsids = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_obsids) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                obsids.extend(batch_obsids.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test results shape:', preds.shape, trues.shape)
        
        if self.label_scaler:
            preds = self.label_scaler.inverse_transform(preds)
            trues = self.label_scaler.inverse_transform(trues)

        # --- Calculate and print metrics ---
        metrics_dict = calculate_metrics(preds, trues, self.targets)
        feh_metrics = calculate_feh_classification_metrics(preds, trues)
        
        print("Test Set Regression Metrics:")
        print(format_metrics(metrics_dict))
        
        print("Test Set FeH Classification Metrics:")
        print(format_feh_classification_metrics(feh_metrics))
        
        # --- Save results to CSV and metrics plots ---
        if hasattr(self.args, 'run_dir') and self.args.run_dir:
            save_dir = self.args.run_dir+'/test_results/'
            os.makedirs(save_dir, exist_ok=True)

            # Create DataFrame for CSV export
            results_df = pd.DataFrame({'obsid': obsids})
            results_df['obsid'] = results_df['obsid'].astype('int64')
            for i, target_name in enumerate(self.targets):
                results_df[f'{target_name}_true'] = trues[:, i]
                results_df[f'{target_name}_pred'] = preds[:, i]
            
            # Reorder columns to have true/pred pairs
            column_order = ['obsid']
            for target_name in self.targets:
                column_order.append(f'{target_name}_true')
                column_order.append(f'{target_name}_pred')
            results_df = results_df[column_order]

            # Save to CSV
            csv_path = os.path.join(save_dir, 'predictions.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Test results saved to {csv_path}")

            # Save metric plots
            save_regression_metrics(metrics_dict, save_dir, self.targets, phase="test")
            save_feh_classification_metrics(feh_metrics, save_dir, phase="test")
        
        return metrics_dict['mae'], metrics_dict['mse'], metrics_dict['rmse']