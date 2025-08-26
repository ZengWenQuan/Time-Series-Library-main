import os
import torch
import torch.nn as nn
import time
import datetime
from utils.tools import EarlyStopping
from utils.augmentations import Transforms
from utils.stellar_metrics import save_history_plot
import mlflow
import logging
import numpy as np
import pandas as pd
import yaml
from utils.stellar_metrics import calculate_metrics, format_metrics, save_regression_metrics, calculate_feh_classification_metrics, save_feh_classification_metrics, format_feh_classification_metrics
from models.registries import MODEL_REGISTRY

def format_duration(seconds):
    """Formats a duration in seconds into a human-readable string (H:M:S)."""
    if seconds < 0:
        return "0s"
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        return f'{minutes}m {seconds}s'
    else:
        return f'{seconds}s'

# --- ADDED: Custom Formatter for Beijing Time ---
class BeijingTimeFormatter(logging.Formatter):
    def converter(self, timestamp):
        import datetime
        tz = datetime.timezone(datetime.timedelta(hours=8))
        return datetime.datetime.fromtimestamp(timestamp, tz)

    def formatTime(self, record, datefmt=None):
        import datetime
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat(timespec='milliseconds')
        return s

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._setup_logger()

        # --- 从 model_conf.yaml 加载设置 ---
        if not hasattr(self.args, 'model_conf') or not self.args.model_conf or not os.path.exists(self.args.model_conf):
            raise FileNotFoundError(f"必须通过 --model_conf 提供一个有效的模型配置文件路径。")

        try:
            with open(self.args.model_conf, 'r') as f:
                model_config = yaml.safe_load(f)
            training_settings = model_config.get('training_settings', {})
            
            # --- 强制从 YAML 文件中获取 'targets' ---
            if 'targets' in training_settings:
                # PyYAML 会自动将 yaml 列表转为 python 列表
                self.targets = training_settings['targets']
                self.args.targets = self.targets  # 保持 args 对象同步
            else:
                raise ValueError(f"配置文件 {self.args.model_conf} 的 'training_settings' 中必须提供 'targets' 键。")

            # --- 加载其他设置（如果不存在则使用默认值） ---
            self.args.loss = training_settings.get('loss_function', 'mae')
            self.args.lradj = training_settings.get('lradj', 'cos')
            self.args.loss_weights = training_settings.get('loss_weights', [1,1,1,1])
            self.args.use_amp = training_settings.get('mixed_precision', True)

        except Exception as e:
            self.logger.error(f"处理模型配置文件 '{self.args.model_conf}' 时出错: {e}")
            raise  # 重新抛出异常，中断程序

        # --- ADDED: Initialize GradScaler for AMP ---
        self.scaler = None
        if getattr(args, 'use_amp', False):
            if self.device.type == 'cuda':
                self.scaler = torch.amp.GradScaler('cuda')
                self.logger.info("Automatic Mixed Precision (AMP) enabled.")
            else:
                self.logger.warning("AMP is only available on CUDA devices. Disabling AMP.")

        # --- ADDED: Resume from checkpoint logic ---
        if getattr(self.args, 'resume_from', None) and os.path.exists(self.args.resume_from):
            self.logger.info(f"Resuming training from checkpoint: {self.args.resume_from}")
            self.model.load_state_dict(torch.load(self.args.resume_from, map_location=self.device))

    def _build_train_transforms(self):
        """从配置文件加载增强配置，创建流水线并直接附加到args。"""
        augs_conf = []
        if hasattr(self.args, 'stats_path') and self.args.stats_path and os.path.exists(self.args.stats_path):
            with open(self.args.stats_path, 'r') as f:
                stats = yaml.safe_load(f)
            augs_conf = stats.get('augs_conf', [])
        
        self.logger.info("Initializing augmentation pipeline and attaching to args...")
        # 将构建好的流水线对象直接赋值给 self.args 的一个新属性
        self.args.train_transform = Transforms(augs_conf)
        # 此函数不再有返回值

    def _build_model(self, sample_batch=None):
        model_class = MODEL_REGISTRY.get(self.args.model)
        if model_class is None:
            raise ValueError(f"Model '{self.args.model}' is not registered. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")
        
        print(f"Building model: {self.args.model}")
        # Add sample_batch to the config object
        self.args.sample_batch = sample_batch
        model = model_class(self.args).float()

        model.to(self.device)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # --- Model Summary Logic ---
        # 模型在自己的__init__中完成初始化，所以这里可以直接计算参数
        info_model_path = os.path.join(self.args.run_dir, 'model.txt')
        with open(info_model_path, 'w') as f:
            f.write("模型结构:\n")
            f.write(f"{model}\n\n")
            
            # --- Per-Submodule Parameter Count ---
            f.write("模块参数量:\n")
            model_to_inspect = model.module if isinstance(model, nn.DataParallel) else model
            
            submodule_attrs = [
                'continuum_branch',
                'normalized_branch',
                'fusion',
                'prediction_head'
            ]
            
            # Check for submodule existence and count their parameters
            for attr in submodule_attrs:
                if hasattr(model_to_inspect, attr):
                    submodule = getattr(model_to_inspect, attr)
                    if isinstance(submodule, nn.Module):
                        submodule_params = sum(p.numel() for p in submodule.parameters())
                        if submodule_params > 0:
                            f.write(f"  - {attr}: {submodule_params:,} 参数\n")
            
            # --- ADDED: Write Overall Complexity ---
            f.write("\n--- 模型复杂度 ---")
            if hasattr(model_to_inspect, 'params') and model_to_inspect.params > 0:
                f.write(f"总参数量: {model_to_inspect.params / 1e6:.2f} M\n")
            else: # Fallback if params not pre-calculated
                total_params = sum(p.numel() for p in model.parameters())
                f.write(f"总参数量: {total_params / 1e6:.2f} M\n")

            if hasattr(model_to_inspect, 'flops') and model_to_inspect.flops > 0:
                f.write(f"GFLOPs: {model_to_inspect.flops / 1e9:.2f}\n")
            f.write("------------------\n\n")

            # --- Per-Layer Parameter Count (Original Logic) ---
            f.write("每层详细参数:\n")
            for name, param in model.named_parameters():
                f.write(f"  {name}: {param.numel():,} 参数\n")

        return model

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        self.train_data, self.train_loader =None
        self.vali_data, self.vali_loader = None
        self.test_data, self.test_loader =None
        raise NotImplementedError("Subclasses must implement _get_data()")

    def _select_optimizer(self):
        # --- Optimizer Selection ---
        # Instructions: Uncomment the optimizer you want to use.
        import torch.optim as optim
        # Option 1: Adam (Default)
        # optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        # Option 2: AdamW
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        # Option 3: SGD with Momentum
        # momentum = getattr(self.args, 'momentum', 0.9) # You can set momentum in your YAML if needed
        # optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=momentum, weight_decay=self.args.weight_decay)
        
        # Option 4: RMSprop
        # optimizer = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        return optimizer

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'cos':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    def get_feature_scaler(self):
        raise NotImplementedError("Subclasses must implement get_feature_scaler()")
    def get_label_scaler(self):
        raise NotImplementedError("Subclasses must implement get_label_scaler()")
    
    def calculate_and_save_all_metrics(self, preds, trues, phase, save_as):
        pass 
    def trace_metrics(self, epoch, train_metrics=None, vali_metrics=None, test_metrics=None, combined_metrics=None):
        """
        将每个数据集中、每个具体标签的MAE指标追踪到MLflow。
        """
        metrics_sets = [
            ("train", train_metrics),
            ("val", vali_metrics),
            ("test", test_metrics),
            ("combined", combined_metrics)
        ]
        # self.logger.info(f"Epoch {epoch + 1}: Logging per-label MAE to MLflow...")
        for phase, metrics_dict in metrics_sets:
            if metrics_dict is None: continue
            for key, value in metrics_dict.items():
                if key.endswith('_mae') and key != 'mae':
                    mlflow_key = f"{phase}_{key}"
                    mlflow.log_metric(mlflow_key, value, step=epoch)
    def vali(self, vali_data, vali_loader, criterion):
        if not vali_data: return None, None, None ,None
        total_loss, all_preds, all_trues,all_obsids = [], [], [],[]
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_obsid) in enumerate(vali_loader):
                outputs = self.model(batch_x.float().to(self.device))
                pred, true = outputs.detach(), batch_y.float().detach()
                loss = criterion(pred.to(self.device), true.to(self.device))
                total_loss.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_trues.append(true.cpu().numpy())
                all_obsids.append(batch_obsid.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        all_obsids=np.concatenate(all_obsids,axis=0)
        if self.label_scaler: 
            all_preds = self.label_scaler.inverse_transform(all_preds)
            all_trues = self.label_scaler.inverse_transform(all_trues)
        return np.average(total_loss), all_preds, all_trues,all_obsids
        raise NotImplementedError("Subclasses must implement vali()")
    def train(self):
        # mlflow.set_experiment(self.args.task_name)
        # mlflow.start_run(run_name=f"{self.args.model}_{self.args.model_id}")
        # mlflow.log_params(vars(self.args))

        chechpoint_path=os.path.join(self.args.run_dir, 'checkpoints')
        os.makedirs(chechpoint_path, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()
        # mlflow.log_param("loss_function", criterion.__class__.__name__)

        history_train_loss, history_vali_loss, history_lr = [], [], []
        best_feh_mae = float('inf') # ADDED: Initialize tracker
        epoch_time=time.time()
        for epoch in range(self.args.train_epochs):
            epoch_grad_norms = [] #<-- ADDED
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_obsid) in enumerate(self.train_loader):
                model_optim.zero_grad() 
                
                # --- ADDED: AMP Logic ---
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_x.float().to(self.device))
                        #loss = sum(criterion(outputs[:, i], batch_y[:, i].float().to(self.device)) * self.args.loss_weights[i]/sum(self.args.loss_weights) for i in range(outputs.shape[1])) if hasattr(self.args, 'loss_weights') and self.args.loss_weights and len(self.args.loss_weights) == outputs.shape[1] else criterion(outputs, batch_y.float().to(self.device))
                        loss=criterion(outputs, batch_y.float().to(self.device))
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(model_optim)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    # Standard training
                    outputs = self.model(batch_x.float().to(self.device))
                    #loss = sum(criterion(outputs[:, i], batch_y[:, i].float().to(self.device)) * self.args.loss_weights[i]/sum(self.args.loss_weights) for i in range(outputs.shape[1])) if hasattr(self.args, 'loss_weights') and self.args.loss_weights and len(self.args.loss_weights) == outputs.shape[1] else criterion(outputs, batch_y.float().to(self.device))
                    loss=criterion(outputs, batch_y.float().to(self.device))
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                    model_optim.step()
                
                epoch_grad_norms.append(grad_norm.item())
                train_loss.append(loss.item())

            # --- Evaluation ---
            train_loss_avg = np.average(train_loss)
            vali_loss, vali_preds, vali_trues,_ = self.vali(self.vali_data, self.vali_loader, criterion)

            if self.test_loader:
                test_loss, test_preds, test_trues ,_ = self.vali(self.test_data, self.test_loader, criterion)
            else:
                test_loss, test_preds, test_trues = None, None, None

            train_eval_loss, train_preds, train_trues ,_ = self.vali(self.train_data, self.train_loader, criterion)

            # --- Metric Processing ---
            train_reg_metrics = self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "latest")
            vali_reg_metrics = self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "latest")
            if self.test_loader:
                test_reg_metrics = self.calculate_and_save_all_metrics(test_preds, test_trues, "test", "latest")
            
            if test_preds is not None:
                combined_preds = np.concatenate([vali_preds, test_preds])
                combined_trues = np.concatenate([vali_trues, test_trues])
                combined_reg_metrics = self.calculate_and_save_all_metrics(combined_preds, combined_trues, "combined_val_test", "latest")

            # --- Logging and History ---
            avg_grad_norm = np.mean(epoch_grad_norms)
            cost_time = time.time() - epoch_time
            epoch_time = time.time()
            remaining_time = cost_time * (self.args.train_epochs - epoch - 1)

            current_lr = model_optim.param_groups[0]['lr']
            history_train_loss.append(train_loss_avg); history_vali_loss.append(vali_loss); history_lr.append(current_lr)
            
            formatted_cost_time = format_duration(cost_time)
            formatted_eta = format_duration(remaining_time)

            log_msg = f"Epoch: {epoch + 1} /{self.args.train_epochs} | Train Loss: {train_loss_avg:.4f} | Vali Loss: {vali_loss:.4f}"
            if test_loss is not None: log_msg += f" | Test Loss: {test_loss:.4f}"
            log_msg += f" | Grad: {avg_grad_norm:.4f} | LR: {current_lr:.6f}"
            log_msg += f" | Time: {formatted_cost_time} | ETA: {formatted_eta}"
            self.logger.info(log_msg)
            
            # --- Log to MLflow ---
            if (epoch + 1) % self.args.vali_interval == 0:
                #self.logger.info(f"--- Detailed Metrics @ Epoch {epoch + 1} ---")
                if train_reg_metrics: print(f"train Metrics:\n{format_metrics(train_reg_metrics)}")
                if vali_reg_metrics: print(f"Validation Metrics:\n{format_metrics(vali_reg_metrics)}")
                if locals().get('test_reg_metrics',None): print(f"Test Metrics:\n{format_metrics(test_reg_metrics)}")
                if 'combined_reg_metrics' in locals() and combined_reg_metrics is not None: print(f"Combined Val/Test Metrics:\n{format_metrics(combined_reg_metrics)}")

            # --- Early Stopping & Best Model Checkpoint Saving ---
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                self.logger.info(f"New best validation loss: {vali_loss:.4f}. Model checkpoint saved.")

            # --- User's Logic: Save metric files if combined FeH_mae is the best so far ---
            if 'combined_reg_metrics' in locals() and combined_reg_metrics and 'FeH_mae' in combined_reg_metrics:
                current_feh_mae = combined_reg_metrics['FeH_mae']
            else:
                current_feh_mae= vali_reg_metrics['FeH_mae']
            if current_feh_mae < best_feh_mae:
                best_feh_mae = current_feh_mae
                torch.save(self.model.state_dict(), chechpoint_path + '/' + 'best.pth')

                self.logger.info(f"New best  FeH MAE: {best_feh_mae:.4f}. Saving metrics for this epoch as 'best'...")
                self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "best")
                self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "best")
                if test_preds is not None:
                    self.calculate_and_save_all_metrics(test_preds, test_trues, "test", "best")
                    self.calculate_and_save_all_metrics(combined_preds, combined_trues, "combined_val_test", "best")

            if early_stopping.early_stop: break
            if scheduler is not None: scheduler.step() 

            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir)
        # mlflow.log_artifact(self.args.run_dir, artifact_path="results")
        # mlflow.end_run()
        return self.model


    def test(self):
        # 1. 加载最优模型
        checkpoint_path = os.path.join(self.args.run_dir, 'checkpoints', 'best.pth')
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading best model from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            self.logger.warning("No best model checkpoint found. Testing on the final model state.")

        # 2. 创建保存目录
        save_dir = os.path.join(self.args.run_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        # 3. 在测试集上评估
        self.logger.info("--- Starting Final Test ---")
        if self.test_data:
            test_loss, test_preds, test_trues ,_= self.vali(self.test_data,self.test_loader, self._select_criterion())
        else:
            test_loss, test_preds, test_trues ,_= self.vali(self.vali_data,self.vali_loader, self._select_criterion())

        if test_preds is None:
            self.logger.warning("Test evaluation returned no results. Skipping metric calculation and saving.")
            return

        # 4. 保存预测值和真实值到CSV
        self.logger.info(f"Saving predictions to {save_dir}")
        pred_df_data = {}
        for i, target_name in enumerate(self.targets):
            pred_df_data[f'{target_name}_true'] = test_trues[:, i]
            pred_df_data[f'{target_name}_pred'] = test_preds[:, i]
        
        pred_df = pd.DataFrame(pred_df_data)
        pred_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

        # 5. 计算并保存所有指标
        self.logger.info("Calculating and saving final metrics...")
        reg_metrics = calculate_metrics(test_preds, test_trues, self.targets)
        cls_metrics = calculate_feh_classification_metrics(test_preds, test_trues, self.args.feh_index)
        
        save_regression_metrics(reg_metrics, save_dir, self.args.targets, phase="final_test")
        save_feh_classification_metrics(cls_metrics, save_dir, phase="final_test")
        
        # 如果MLflow仍在运行，记录最终测试指标
        # if mlflow.active_run():
        #     mlflow.log_metric("final_test_loss", test_loss)
        #     for metric_name, metric_value in reg_metrics.items():
        #         if isinstance(metric_value, (int, float)):
        #             mlflow.log_metric(f'final_test_{metric_name}', metric_value)
        #     mlflow.log_artifacts(save_dir, artifact_path="test_results")

    def test_all(self):
        # 1. 加载最优模型
        checkpoint_path = self.args.checkpoints
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            self.logger.error(f"No valid checkpoint path provided via --checkpoints. Aborting test_all.")
            return
        self.logger.info(f"Loading model from provided checkpoint: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # --- Loop through each dataset (train, val, test) and evaluate ---
        dataset_splits = {
            'train': (self.train_data, self.train_loader),
            'val': (self.vali_data, self.vali_loader),
            'test': (self.test_data, self.test_loader)
        }

        for split_name, (data, loader) in dataset_splits.items():
            if data is None or loader is None:
                self.logger.warning(f"No data/loader for '{split_name}' split. Skipping.")
                continue

            self.logger.info(f"--- Starting evaluation on {split_name} data ---")
            
            # 2. 创建保存目录
            save_dir = os.path.join(self.args.run_dir, 'test_all_results', split_name)
            os.makedirs(save_dir, exist_ok=True)

            # 3. 在当前数据集上评估
            loss, preds, trues, obsids = self.vali(data, loader, self._select_criterion())

            if preds is None:
                self.logger.warning(f"Evaluation on '{split_name}' returned no results. Skipping.")
                continue

            # 4. 保存预测值和真实值到CSV
            self.logger.info(f"Saving predictions for '{split_name}' to {save_dir}")
            pred_df = pd.DataFrame({'obsid': obsids})
            for i, target_name in enumerate(self.targets):
                pred_df[f'{target_name}_true'] = trues[:, i]
                pred_df[f'{target_name}_pred'] = preds[:, i]
            pred_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

            # 5. 计算并保存所有指标
            self.logger.info(f"Calculating and saving metrics for '{split_name}'...")
            reg_metrics = calculate_metrics(preds, trues, self.targets)
            cls_metrics = calculate_feh_classification_metrics(preds, trues, self.args.feh_index)
            
            save_regression_metrics(reg_metrics, save_dir, self.args.targets, phase=f"final_{split_name}")
            save_feh_classification_metrics(cls_metrics, save_dir, phase=f"final_{split_name}")

        self.logger.info("--- test_all completed ---")

    def _select_criterion(self):
        loss=self.args.loss.lower()
        if loss == 'Mse'.lower() or loss == 'l2'.lower() :
            return nn.MSELoss()
        elif loss == 'Mae'.lower() or loss =='l1'.lower():
            return nn.L1Loss()
        elif loss == 'SmoothL1'.lower():
            return nn.SmoothL1Loss()
        elif loss == 'Huber'.lower():
            return nn.HuberLoss(delta=1.0)
        elif loss == 'LogCosh'.lower():
            def logcosh_loss(pred, target):
                return torch.mean(torch.log(torch.cosh(pred - target)))
            return logcosh_loss
        else:
            print(f"警告: 未知的损失函数 '{loss}'，使用默认的MSE损失")
            return nn.MSELoss()        
    def _setup_logger(self):
        import datetime
        # Prevent the logger from propagating to the root logger 
        self.logger = logging.getLogger('CEMP_search')        
        self.logger.setLevel(logging.INFO)   
        self.logger.propagate = False
        
        # Formatter - 使用 %(asctime)s 来动态显示时间
        formatter = BeijingTimeFormatter(
            'CEMP search - %(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S' # 可选：定义时间格式
        )
        
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