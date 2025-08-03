from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
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

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils.stellar_metrics import Scaler

warnings.filterwarnings('ignore')


class Exp_Spectral_Prediction(Exp_Basic):
    """
    恒星参数估计（Stellar Parameter Estimation）实验类
    """
    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)
        # 恒星参数名称
        self.targets = args.targets
        self.args=args
        self.label_scaler=self.get_label_scaler()
        self.feature_scaler=self.get_feature_scaler()
        info_model=self.args.run_dir+'/model.txt'
        with open(info_model,'w') as f:
            # 写入模型结构
            f.write("模型结构:\n")
            f.write(f"{self.model}\n\n")
            
            # 写入每层参数数量
            f.write("每层参数数量:\n")
            for name, param in self.model.named_parameters():
                f.write(f"  {name}: {param.numel():,} 参数\n")
    def _build_model(self):
        self.args.enc_in = self.args.feature_size * 2
        self.args.c_out = self.args.label_size
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.label_scaler, self.feature_scaler)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


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
        
        return np.average(total_loss) , all_preds , all_trues

    def train(self,setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        chechpoint_path=self.args.run_dir+'/'+'checkpoints'
        if not os.path.exists(chechpoint_path):
            os.makedirs(chechpoint_path)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            grad_norm_before = 0
            grad_norm_after = 0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_obsid) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                # Check for abnormally high loss
                if hasattr(self.args, 'loss_threshold') and loss.item() > self.args.loss_threshold:
                    print(f"[Warning] Batch {i+1}/{train_steps}: Loss {loss.item():.2f} exceeds threshold {self.args.loss_threshold}. Skipping batch.")
                    continue # Skip the rest of the loop for this batch

                train_loss.append(loss.item())
                loss.backward()

                # --- Gradient Clipping ---
                max_norm = self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 1.0
                
                # Calculate norm before clipping and perform clipping
                grad_norm_before_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                if torch.isfinite(grad_norm_before_tensor):
                    grad_norm_before = grad_norm_before_tensor.item()
                else:
                    grad_norm_before = float('inf')

                # Calculate norm after clipping
                norm_after_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        norm_after_sq += param_norm.item() ** 2
                grad_norm_after = norm_after_sq ** 0.5
                # --- End Gradient Clipping ---

                model_optim.step()

            train_loss = np.average(train_loss)
            
            do_validation = (epoch + 1) % self.args.vali_interval == 0
            is_last_epoch = epoch == self.args.train_epochs - 1

            vali_loss, vali_preds, vali_trues = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(test_data, test_loader, criterion)
            
            left_time = (self.args.train_epochs - epoch) *(time.time() - epoch_time)
            
            grad_info = f"Grad Norm: {grad_norm_before:.2f}->{grad_norm_after:.2f}"

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss),end=" ")
            if test_data is not None:
                print("Test Loss: {0:.7f}".format(test_loss),end=" ")
            print("Vali Interval: {0},left_time: {1:.2f}h{2:.2f}m{3:.2f}s | {4}".format(
                self.args.vali_interval, left_time//3600, (left_time%3600)//60, left_time%60, grad_info))
            
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

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        last_model_path = chechpoint_path + '/' + 'last.pth'
        torch.save(self.model.state_dict(), last_model_path)
        print("the train is over,save the best model and the last model")
        return self.model

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