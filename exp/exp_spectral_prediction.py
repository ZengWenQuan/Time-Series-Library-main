from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.spectral_prediction import DualBranchMoENet

from utils.tools import EarlyStopping
from utils.stellar_metrics import calculate_metrics, format_metrics, save_regression_metrics, save_feh_classification_metrics, save_history_plot,save_feh_classification_metrics,calculate_feh_classification_metrics
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
        # 从配置文件加载额外设置
        if hasattr(args, 'model_conf') and args.model_conf and os.path.exists(args.model_conf):
            try:
                with open(args.model_conf, 'r') as f:
                    model_config = yaml.safe_load(f)
                
                # 读取训练设置
                training_settings = model_config.get('training_settings', {})
                self.loss_function_name = training_settings.get('loss_function', 'RegressionFocalLoss')

                # 读取混合精度设置
                use_amp_from_conf = model_config.get('mixed_precision', False)
                if use_amp_from_conf and not getattr(args, 'use_amp', False):
                    print("✓ 从模型配置文件中启用混合精度训练")
                    self.use_amp = True
                else:
                    self.use_amp = getattr(args, 'use_amp', False)

            except Exception as e:
                print(f"警告: 无法从配置文件中读取设置: {e}")
                self.loss_function_name = 'RegressionFocalLoss'
                self.use_amp = getattr(args, 'use_amp', False)
        else:
            self.loss_function_name = 'mae'
            self.use_amp = getattr(args, 'use_amp', False)

        # 初始化混合精度 scaler
        if self.use_amp:
            print("✓ 混合精度训练已启用 (AMP)")
            self.scaler = GradScaler()
        else:
            print("✓ 使用标准精度训练")
            self.scaler = None
        
        self.label_scaler=self.get_label_scaler()
        self.feature_scaler=self.get_feature_scaler()
        self._get_data()

    def _select_criterion(self):
        # 1. 最高优先级：如果使用概率头，则强制使用NLL损失
        if hasattr(self.model, 'head_type') and self.model.head_type == 'probabilistic':
            print("✓ 检测到概率预测头，强制使用高斯负对数似然损失 (GaussianNLLLoss)")
            return GaussianNLLLoss()

        # 2. 创建损失函数别名映射 (不区分大小写)
        loss_mapping = {
            'l1': nn.L1Loss,
            'mae': nn.L1Loss,
            'l2': nn.MSELoss,
            'mse': nn.MSELoss,
            'regressionfocalloss': RegressionFocalLoss,
            'gaussiannllloss': GaussianNLLLoss
        }

        # 3. 从配置中获取名称，并进行标准化处理
        normalized_loss_name = self.loss_function_name.lower()

        # 4. 查找并实例化损失函数
        loss_class = loss_mapping.get(normalized_loss_name)

        if loss_class:
            print(f"✓ 使用 {loss_class.__name__} 损失函数 (根据配置 '{self.loss_function_name}')")
            return loss_class()
        else:
            # 5. 如果找不到，则使用默认值并发出警告
            print(f"警告: 未知的损失函数别名 '{self.loss_function_name}'. 将使用默认的 RegressionFocalLoss.")
            return RegressionFocalLoss()

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif self.args.lradj == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        elif self.args.lradj == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif self.args.lradj == 'warmup_cosine':
            # This remains a custom implementation, as it was before.
            return None # Returning None will keep the old behavior
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return scheduler
  
               

    def get_feature_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            
            return Scaler(scaler_type=self.args.features_scaler_type, stats_dict={'flux': stats['flux']}, target_names=['flux'])
        else:
            raise ValueError("没有提供特征统计数据文件路径")

    def get_label_scaler(self):
        if self.args.stats_path:
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)

            return Scaler(scaler_type=self.args.label_scaler_type, stats_dict=stats, target_names=self.targets)
        else:
            raise ValueError("没有提供标签统计数据文件路径")
        
    def vali(self, vali_data, vali_loader, criterion):
        if vali_data is None:
            return None, None, None
        total_loss = []
        self.model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for i, (batch_x_continuum,batch_x_normalized, batch_y,batch_obsid) in enumerate(vali_loader):
                batch_x_continuum = batch_x_continuum.float().to(self.device)
                batch_x_normalized=batch_x_normalized.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x_continuum,batch_x_normalized)
                
                # 处理不同预测头的输出
                if hasattr(self.model, 'head_type') and self.model.head_type == 'probabilistic':
                    pred = outputs[..., 0].detach() # 只取均值用于评估
                    loss = criterion(outputs, batch_y)
                else:
                    pred = outputs.detach()
                    loss = criterion(pred, batch_y)

                true = batch_y.detach()
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

    def _get_data(self):
        
        self.train_data, self.train_loader = data_provider(args=self.args,flag='train', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
        self.vali_data, self.vali_loader = data_provider(args=self.args,flag='val', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)

        self.test_data, self.test_loader = data_provider(args=self.args,flag='test', feature_scaler=self.feature_scaler, label_scaler=self.label_scaler)
    def train(self):
        # --- MLflow Setup ---
        mlflow.set_experiment(self.args.task_name)
        mlflow.start_run(run_name=f"{self.args.model}_{self.args.model_id}")
        mlflow.log_params(vars(self.args))

        chechpoint_path=self.args.run_dir+'/'+'checkpoints'
        if not os.path.exists(chechpoint_path):
            os.makedirs(chechpoint_path)
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()

        final_loss_function_name = criterion.__class__.__name__
        mlflow.log_param("loss_function", final_loss_function_name)
        self.logger.info(f"MLflow: Logged loss_function = {final_loss_function_name}")
        
        # --- Infer Model Signature for MLflow ---
        self.logger.info("Inferring model signature for MLflow...")

        # 两个输入分支都使用由超参数定义的同一个feature_size
        feature_size = self.args.feature_size
        self.logger.info(f"Feature size for each input branch: {feature_size}")

        # 创建两个同样大小的虚拟样本
        input_continuum = torch.randn(1, feature_size).to(self.device)
        input_normalized = torch.randn(1, feature_size).to(self.device)

        # 将两个独立的样本传递给模型
        output_sample = self.model(torch.randn(1, feature_size).to(self.device), torch.randn(1, feature_size).to(self.device))

        if isinstance(output_sample, tuple):
            output_sample = output_sample[0]  # 使用主输出作为签名

        # 为多输入签名创建一个字典
        input_signature_data = {
            "x_continuum": input_continuum.cpu().numpy(),
            "x_normalized": input_normalized.cpu().numpy()
        }

        from mlflow.models.signature import infer_signature
        signature = infer_signature(input_signature_data, output_sample.detach().cpu().numpy())
        self.logger.info("成功推断出包含两个输入的模型签名。 সন")

        history_train_loss = []
        history_vali_loss = []
        history_lr = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_grad_norms_before = []
            epoch_grad_norms_after = []

            self.model.train()
            epoch_time = time.time()
            
            max_norm = self.args.max_grad_norm if hasattr(self.args, 'max_grad_norm') else 20.0
            
            for i, (batch_x_continuum, batch_x_normalized, batch_y, batch_obsid) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_continuum = batch_x_continuum.float().to(self.device)
                batch_x_normalized = batch_x_normalized.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.use_amp and self.scaler is not None:
                    with autocast():
                        model_output = self.model(batch_x_continuum, batch_x_normalized)
                        if isinstance(model_output, tuple) and self.model.training:
                            outputs, aux_loss = model_output
                            main_loss = criterion(outputs, batch_y)
                            loss = main_loss + aux_loss
                        else:
                            outputs = model_output
                            main_loss = criterion(outputs, batch_y)
                            loss = main_loss
                            aux_loss = torch.tensor(0.0)

                    if hasattr(self.args, 'loss_threshold') and loss.item() > self.args.loss_threshold:
                        print(f"[Warning] Batch {i+1}/{train_steps}: Loss {loss.item():.2f} exceeds threshold. Skipping.")
                        continue
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(model_optim)
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm).item()
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    model_output = self.model(batch_x_continuum, batch_x_normalized)
                    if isinstance(model_output, tuple) and self.model.training:
                        outputs, aux_loss = model_output
                        main_loss = criterion(outputs, batch_y)
                        loss = main_loss + aux_loss
                    else:
                        outputs = model_output
                        main_loss = criterion(outputs, batch_y)
                        loss = main_loss
                        aux_loss = None

                    if hasattr(self.args, 'loss_threshold') and loss.item() > self.args.loss_threshold:
                        print(f"[Warning] Batch {i+1}/{train_steps}: Loss {loss.item():.2f} exceeds threshold. Skipping.")
                        continue
                    
                    loss.backward()
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm).item()
                    model_optim.step()

                norm_after_sq = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None)
                grad_norm_after = norm_after_sq ** 0.5
                epoch_grad_norms_before.append(grad_norm_before)
                epoch_grad_norms_after.append(grad_norm_after)

                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    mlflow.log_metric('batch_total_loss', loss.item(), step=epoch * train_steps + i)
                    mlflow.log_metric('batch_main_loss', main_loss.item(), step=epoch * train_steps + i)
                    if aux_loss is not None:
                        mlflow.log_metric('batch_aux_loss', aux_loss.item(), step=epoch * train_steps + i)

            train_loss = np.average(train_loss)
            avg_grad_norm_before = np.mean(epoch_grad_norms_before)
            avg_grad_norm_after = np.mean(epoch_grad_norms_after)
            
            vali_loss, vali_preds, vali_trues = self.vali(self.vali_data, self.vali_loader, criterion)
            test_loss, test_preds, test_trues = self.vali(self.test_data, self.test_loader, criterion)
            
            history_train_loss.append(train_loss)
            history_vali_loss.append(vali_loss)

            current_lr = model_optim.param_groups[0]['lr']
            history_lr.append(current_lr)

            grad_info = f"Grad Norm (Avg): {avg_grad_norm_before:.2f} -> {avg_grad_norm_after:.2f}"
            log_message = f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.4f} | Vali Loss: {vali_loss:.4f}"
            if self.test_data is not None:
                log_message += f" | Test Loss: {test_loss:.4f}"
            log_message += f" | lr: {current_lr:.6f} | {grad_info}"
            self.logger.info(log_message)
            
            metrics_dict_vali = calculate_metrics(vali_preds, vali_trues, self.args.targets)
            if self.test_data is not None:
                metrics_dict_test = calculate_metrics(test_preds, test_trues, self.args.targets)

            # --- ADDED CODE START ---
            # 1. Conditionally print detailed metrics based on interval
            if (epoch + 1) % self.args.vali_interval == 0:
                self.logger.info(f"--- Detailed Metrics @ Epoch {epoch + 1} ---")
                if self.vali_data:
                    self.logger.info(f"Validation Metrics:\n{format_metrics(metrics_dict_vali)}")
                if self.test_data is not None:
                    self.logger.info(f"Test Metrics:\n{format_metrics(metrics_dict_test)}")
            
            # 2. Save latest metrics to files every epoch
            save_regression_metrics(metrics_dict_vali, self.args.run_dir+'/metrics/latest', self.args.targets, phase="val")
            save_regression_metrics(metrics_dict_test, self.args.run_dir+'/metrics/latest', self.args.targets, phase="test")
           
           
            # --- ADDED: Save latest classification metrics ---
            class_metrics_vali = calculate_feh_classification_metrics(vali_preds, vali_trues, self.args.feh_index)
            save_feh_classification_metrics(class_metrics_vali, self.args.run_dir+'/metrics/latest', phase="val")
            class_metrics_test = calculate_feh_classification_metrics(test_preds, test_trues, self.args.feh_index)
            save_feh_classification_metrics(class_metrics_test, self.args.run_dir+'/metrics/latest', phase="test")
            # --- ADDED CODE END ---
                
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                best_model_path = os.path.join(chechpoint_path, 'best.pth')
                self.logger.info(f"Validation loss improved. Logging new best model to{best_model_path} ")
                # --- ADDED CODE START ---
                # 3. Save best metrics to files
                self.logger.info("Saving best metrics to files...")
                save_regression_metrics(metrics_dict_vali, self.args.run_dir+'/metrics/best', self.args.targets, phase="val")
                if self.test_data is not None:
                    save_regression_metrics(metrics_dict_test, self.args.run_dir+'/metrics/best', self.args.targets, phase="test")
                # --- ADDED: Save best classification metrics ---
                # Note: metrics were already calculated for the 'latest' save, so we reuse them here.
                save_feh_classification_metrics(class_metrics_vali, self.args.run_dir+'/metrics/best', phase="val")
                if self.test_data is not None:
                    save_feh_classification_metrics(class_metrics_test, self.args.run_dir+'/metrics/best', phase="test")
                # --- ADDED CODE END ---
                torch.save(self.model.state_dict(), best_model_path)

            latest_model_path = os.path.join(chechpoint_path, 'latest.pth')
            torch.save(self.model.state_dict(), latest_model_path)

            mlflow.log_metric('train_loss', train_loss, step=epoch)
            if vali_loss is not None:
                mlflow.log_metric('val_loss', vali_loss, step=epoch)
            if test_loss is not None:
                mlflow.log_metric('test_loss', test_loss, step=epoch)
            mlflow.log_metric('learning_rate', current_lr, step=epoch)

            if metrics_dict_vali:
                for metric_name, metric_value in metrics_dict_vali.items():
                    if 'mae' in metric_name.lower():
                        mlflow.log_metric(f'val_{metric_name}', metric_value, step=epoch)
            
            if self.test_data is not None and metrics_dict_test:
                for metric_name, metric_value in metrics_dict_test.items():
                    if 'mae' in metric_name.lower():
                        mlflow.log_metric(f'test_{metric_name}', metric_value, step=epoch)

            # --- ADDED: Update and save history plot every epoch ---
            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir)

            # --- ADDED: Update and save history plot every epoch ---
            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if scheduler is not None:
                scheduler.step()
            
        mlflow.log_artifact(latest_model_path, artifact_path="checkpoints")
        mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
        mlflow.end_run()

        return self.model

    def test(self, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.run_dir, 'checkpoints', 'best.pth')))

        preds = []
        trues = []
        obsids = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_continuum, batch_x_normalized, batch_y, batch_obsids) in enumerate(self.test_loader):
                batch_x_continuum = batch_x_continuum.float().to(self.device)
                batch_x_normalized = batch_x_normalized.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x_continuum, batch_x_normalized)

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

        metrics_dict = calculate_metrics(preds, trues, self.targets)
        print("Test Set Regression Metrics:")
        print(format_metrics(metrics_dict))
        
        if hasattr(self.args, 'run_dir') and self.args.run_dir:
            save_dir = self.args.run_dir+'/test_results/'
            os.makedirs(save_dir, exist_ok=True)

            results_df = pd.DataFrame({'obsid': obsids})
            results_df['obsid'] = results_df['obsid'].astype('int64')
            for i, target_name in enumerate(self.targets):
                results_df[f'{target_name}_true'] = trues[:, i]
                results_df[f'{target_name}_pred'] = preds[:, i]
            
            column_order = ['obsid']
            for target_name in self.targets:
                column_order.append(f'{target_name}_true')
                column_order.append(f'{target_name}_pred')
            results_df = results_df[column_order]

            csv_path = os.path.join(save_dir, 'predictions.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Test results saved to {csv_path}")

            save_regression_metrics(metrics_dict, save_dir, self.targets, phase="test")
        
        return metrics_dict['mae'], metrics_dict['mse'], metrics_dict['rmse']