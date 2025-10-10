from utils.loss import select_criterion
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

        # --- Load, merge, and re-save model configurations ---
        if not hasattr(self.args, 'model_conf') or not self.args.model_conf or not os.path.exists(self.args.model_conf):
            raise FileNotFoundError("A valid model configuration file path must be provided via --model_conf.")

        try:
            # 1. Load and merge all YAML files into one dictionary
            self.model_config = self._load_and_merge_configs(self.args.model_conf)
            print(self.model_config)
            # 2. Save the merged dictionary to a new temporary file inside the run directory
            merged_config_path = os.path.join(self.args.run_dir,f'{ self.model_config["name"]}.yaml')
            with open(merged_config_path, 'w') as f:
                yaml.dump(self.model_config, f, sort_keys=False, default_flow_style=False, indent=2)
            
            # 3. CRITICAL: Overwrite args.model_conf to point to the new, complete file
            self.args.model_conf = merged_config_path
            self.logger.info(f"Modular configs merged and saved to: {merged_config_path}")

            # 4. Populate args with the full config for other potential uses
            for key, value in self.model_config.items():
                if not hasattr(self.args, key):
                    setattr(self.args, key, value)

            # 5. Continue with setup using the merged config
            training_settings = self.model_config.get('training_settings', {})
            
            if 'targets' in training_settings:
                self.targets = training_settings['targets']
                self.args.targets = self.targets
                self.model_config['targets']=self.targets
            else:
                raise ValueError(f"The 'targets' key must be provided in the training_settings.")

            self.args.loss = training_settings.get('loss_function', 'mse')
            self.args.lradj = training_settings.get('lradj', 'cos')
            self.args.use_amp = training_settings.get('mixed_precision', True)

        except Exception as e:
            self.logger.error(f"Error processing model configuration '{self.args.model_conf}': {e}")
            raise

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
            self.model.load_state_dict(torch.load(self.args.resume_from, map_location=self.device), strict=False)
        self._build_train_transforms()
    def _load_and_merge_configs(self, main_config_path):
        """
        Loads the main YAML config and dynamically merges sub-configs based on module names.
        """
        self.logger.info(f"Loading main configuration from: {main_config_path}")
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)

        # Define the mapping from the name in the main config to the subdirectory and the final key
        module_mapping = {
            'training_name': ('training', 'training_settings'),
            'backbone_name': ('backbone', 'backbone_config'),
            'global_branch_name': ('global_branch', 'global_branch_config'),
            'local_branch_name': ('local_branch', 'local_branch_config'),
            'fusion_name': ('fusion', 'fusion_config'),
            'head_name': ('head', 'head_config'),
        }

        for name_key, (subdir, config_key) in module_mapping.items():
            if name_key in main_config:
                module_name = main_config[name_key]
                base_conf_dir = 'conf' # Always search from the root conf directory
                sub_config_path = os.path.join(base_conf_dir, subdir, f"{module_name}.yaml")
                
                if not os.path.exists(sub_config_path):
                    raise FileNotFoundError(f"Sub-configuration file not found: {sub_config_path}")
                
                self.logger.info(f"Loading sub-config for '{config_key}' from: {sub_config_path}")
                with open(sub_config_path, 'r') as f:
                    sub_config = yaml.safe_load(f)
                
                main_config[config_key] = sub_config
        
        return main_config

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
        model_class = MODEL_REGISTRY.get(self.model_config['name'])
        if model_class is None:
            raise ValueError(f"Model '{self.args.model}' is not registered. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")
        self.model_config['task_name']=self.args.task_name
        self.logger.info(f"Building model: {self.model_config['name']}")
        # Add sample_batch to the config object, which is used by the model's __init__
        self.args.sample_batch = sample_batch
        model = model_class(self.model_config).float()
        model.to(self.device)

        # --- ADDED: Load pre-trained model for finetuning ---
        if self.args.checkpoints and os.path.exists(self.args.checkpoints):
            self.logger.info(f"Loading pre-trained model from: {self.args.checkpoints}")
            try:
                model.load_state_dict(torch.load(self.args.checkpoints, map_location=self.device), strict=False)
            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")

        # --- GFLOPs and Parameters Calculation ---
        model_to_inspect = model.module if isinstance(model, nn.DataParallel) else model
        
        # --- ADDED: Per-submodule profiling ---
        submodule_stats = {}
        if sample_batch is not None and hasattr(model_to_inspect, 'profile_model'):
            # Profile requires input on the same device as the model
            sample_batch_device = sample_batch[0].to(self.device) if isinstance(sample_batch, list) else sample_batch.to(self.device)
            submodule_stats = model_to_inspect.profile_model(sample_batch_device)

        if sample_batch is not None:
            from thop import profile
            sample_batch_device = sample_batch[0].to(self.device) if isinstance(sample_batch, list) else sample_batch.to(self.device)
            macs, params = profile(model_to_inspect, inputs=(sample_batch_device,), verbose=False)
            model_to_inspect.flops = macs * 2
            model_to_inspect.params = params
            self.logger.info(f"Total FLOPs: {model_to_inspect.flops:,.0f}, Total Params: {model_to_inspect.params:,}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # --- Model Summary Logic ---
        info_model_path = os.path.join(self.args.run_dir, 'model.txt')
        with open(info_model_path, 'w') as f:
            f.write("模型结构:\n")
            f.write(f"{model}\n\n")
            
            # --- Per-Submodule Parameter Count & FLOPs ---
            f.write("--- 子模块参数量与FLOPs ---")
            if submodule_stats:
                total_submodule_flops = 0
                total_submodule_params = 0
                for name, stats in submodule_stats.items():
                    params_val = stats.get('params', 0)
                    flops_val = stats.get('flops', 0)
                    total_submodule_params += params_val
                    total_submodule_flops += flops_val
                    params_str = f"{params_val:,}"
                    flops_str = f"{int(flops_val):,}"
                    f.write(f"  - {name}: {params_str} Params, {flops_str} FLOPs\n")
                f.write(f"  - Submodule Total: {total_submodule_params:,} Params, {int(total_submodule_flops):,} FLOPs\n")

            else: # Fallback to old method if profiling function doesn't exist or failed
                f.write("  (Profiling function not available or failed, using parameter count fallback)")
                submodule_attrs = ['backbone', 'global_branch', 'local_branch', 'fusion', 'prediction_head']
                for attr in submodule_attrs:
                    if hasattr(model_to_inspect, attr):
                        submodule = getattr(model_to_inspect, attr)
                        if isinstance(submodule, nn.Module):
                            submodule_params = sum(p.numel() for p in submodule.parameters())
                            if submodule_params > 0:
                                f.write(f"  - {attr}: {submodule_params:,} Params\n")
            
            # --- Overall Complexity ---
            f.write("\n--- 模型复杂度 (总计)---")
            total_params = getattr(model_to_inspect, 'params', sum(p.numel() for p in model.parameters()))
            f.write(f"总参数量: {total_params:,}\n")

            if hasattr(model_to_inspect, 'flops'):
                f.write(f"总FLOPs: {int(model_to_inspect.flops):,}\n")
            f.write("------------------\n\n")

            # --- Per-Layer Parameter Count ---
            f.write("每层详细参数:")
            for name, param in model.named_parameters():
                f.write(f"  {name}: {param.numel():,} 参数\n")

        return model

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                # The primary device is set based on the first ID in the list
                device = torch.device(f'cuda:{self.args.device_ids[0]}')
                print(f'Use Multi-GPU: devices {self.args.devices}. Main device: {device}')
            else:
                # When selecting a single GPU (e.g., GPU 1), we make it the only visible device.
                # PyTorch then sees this single device as 'cuda:0'.
                #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device(f'cuda:{self.args.gpu}')  # Always use cuda:0 when only one device is visible
                print(f'Use GPU: physical device {self.args.gpu} (mapped to cuda:0)')
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
        raise NotImplementedError("Subclasses must implement _get_data()")

    def _get_finetune_data(self):
        raise NotImplementedError("Subclasses must implement _get_finetune_data() to be able to finetune.")

    def _select_optimizer(self):
        # --- Optimizer Selection ---
        import torch.optim as optim

        if getattr(self.args, 'freeze_body', False):
            self.logger.info("Freezing model body, only training the head.")
            
            # First, make sure we're dealing with the base model if it's wrapped
            model_to_inspect = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

            params_to_update = []
            for name, param in model_to_inspect.named_parameters():
                if 'prediction_head' in name or 'head' in name: # More robust check
                    param.requires_grad = True
                    params_to_update.append(param)
                    self.logger.info(f"Parameter '{name}' will be trained.")
                else:
                    param.requires_grad = False
            
            if not params_to_update:
                self.logger.warning("Warning: No parameters found for the prediction head. The optimizer will have nothing to train.")

        else:
            self.logger.info("Training all model parameters.")
            params_to_update = self.model.parameters()

        # Select the optimizer type
        optimizer_type = getattr(self.args, 'optimizer', 'Adam').lower()
        
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(params_to_update, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif optimizer_type == 'sgd':
            momentum = getattr(self.args, 'momentum', 0.9)
            optimizer = optim.SGD(params_to_update, lr=self.args.learning_rate, momentum=momentum, weight_decay=self.args.weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(params_to_update, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else: # Default to Adam
            optimizer = optim.Adam(params_to_update, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        self.logger.info(f"Optimizer selected: {optimizer.__class__.__name__}")
        return optimizer

    def _select_scheduler(self, optimizer):
        if self.args.lradj == 'cos':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
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

            train_eval_loss, train_preds, train_trues ,_ = self.vali(self.train_data, self.train_loader, criterion)

            # --- Metric Processing ---
            train_reg_metrics = self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "latest")
            vali_reg_metrics = self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "latest")
            
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
            log_msg += f" | Grad: {avg_grad_norm:.4f} | LR: {current_lr:.6f}"
            log_msg += f" | Time: {formatted_cost_time} | ETA: {formatted_eta}"
            
            # --- Log to MLflow ---
            if (epoch + 1) % self.args.vali_interval == 0:
                #self.logger.info(f"--- Detailed Metrics @ Epoch {epoch + 1} ---")
                if train_reg_metrics: print(f"train Metrics:\n{format_metrics(train_reg_metrics)}")
                if vali_reg_metrics: print(f"Validation Metrics:\n{format_metrics(vali_reg_metrics)}")

            # --- Early Stopping & Best Model Checkpoint Saving ---
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                self.logger.info(f"New best validation loss: {vali_loss:.4f}. Model checkpoint saved.")

            # --- User's Logic: Save metric files if combined FeH_mae is the best so far ---
            current_feh_mae= vali_reg_metrics['FeH_mae']
            if current_feh_mae < best_feh_mae:
                best_feh_mae = current_feh_mae
                # 清理state_dict，移除thop等分析工具可能添加的非参数键
                state_dict = self.model.state_dict()
                clean_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
                torch.save(clean_state_dict, chechpoint_path + '/' + 'best.pth')

                self.logger.info(f"New best  FeH MAE: {best_feh_mae:.4f}. Saving metrics for this epoch as 'best'...")
                self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "best")
                self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "best")

            if early_stopping.early_stop: break
            if scheduler is not None: scheduler.step() 

            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir)
            self.logger.info(log_msg)
        # mlflow.log_artifact(self.args.run_dir, artifact_path="results")
        # mlflow.end_run()
        return self.model

    def finetune(self, finetune_lr=None, finetune_epochs=None):
        # --- Get Finetuning Data ---
        self._get_finetune_data()

        # --- Setup for Finetuning ---
        original_run_dir = self.args.run_dir
        finetune_dir = os.path.join(original_run_dir, 'finetune')
        self.args.run_dir = finetune_dir # Temporarily switch run directory
        os.makedirs(finetune_dir, exist_ok=True)
        self.logger.info(f"--- Starting Finetuning ---")
        self.logger.info(f"Finetuning outputs will be saved in: {finetune_dir}")

        # --- ADDED: Evaluate model on full dataset BEFORE finetuning ---
        self.logger.info("--- Starting evaluation on full dataset before finetuning ---")
        
        # Store current run_dir and set a new one for saving pre-finetune results
        original_finetune_dir = self.args.run_dir
        pre_finetune_save_dir = os.path.join(original_finetune_dir, 'pre_finetune_evaluation')
        os.makedirs(pre_finetune_save_dir, exist_ok=True)
        self.args.run_dir = pre_finetune_save_dir
        self.logger.info(f"Saving pre-finetuning evaluation results to: {pre_finetune_save_dir}")

        criterion = self._select_criterion()
        
        # --- Evaluate on Train Set ---
        if self.train_loader:
            _, train_preds, train_trues, _ = self.vali(self.train_data, self.train_loader, criterion)
            self.logger.info("Calculating and saving metrics for TRAIN set before finetuning...")
            self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "pre_finetune")

        # --- Evaluate on Validation Set ---
        if self.vali_loader:
            _, vali_preds, vali_trues, _ = self.vali(self.vali_data, self.vali_loader, criterion)
            self.logger.info("Calculating and saving metrics for VALIDATION set before finetuning...")
            self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "pre_finetune")

        # Restore original finetune run_dir
        self.args.run_dir = original_finetune_dir
        self.logger.info(f"Restored run directory to: {self.args.run_dir}")
        # --- END of pre-finetuning evaluation ---

        # --- Check for Finetuning Data ---
        if not all(hasattr(self, attr) for attr in ['finetune_train_loader', 'finetune_vali_data', 'finetune_vali_loader']):
            self.logger.error("Finetuning data not found. `_get_finetune_data()` did not load the required data.")
            self.args.run_dir = original_run_dir # Restore dir before exiting
            return

        # --- Training Setup ---
        chechpoint_path = os.path.join(self.args.run_dir, 'checkpoints')
        os.makedirs(chechpoint_path, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # Set finetuning learning rate
        self.args.learning_rate = self.args.finetune_lr
        model_optim = self._select_optimizer()

        scheduler = self._select_scheduler(model_optim)
        criterion = self._select_criterion()

        history_train_loss, history_vali_loss, history_lr = [], [], []
        best_feh_mae = float('inf')
        epoch_time = time.time()
        
        epochs = self.args.finetune_epochs

        # --- Finetuning Loop ---
        for epoch in range(epochs):
            epoch_grad_norms = []
            self.model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_obsid) in enumerate(self.finetune_train_loader): # Use finetune loader
                model_optim.zero_grad()
                
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_x.float().to(self.device))
                        loss = criterion(outputs, batch_y.float().to(self.device))
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(model_optim)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_x.float().to(self.device))
                    loss = criterion(outputs, batch_y.float().to(self.device))
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                    model_optim.step()
                
                epoch_grad_norms.append(grad_norm.item())
                train_loss.append(loss.item())

            # --- Evaluation ---
            train_loss_avg = np.average(train_loss)
            vali_loss, vali_preds, vali_trues, _ = self.vali(self.finetune_vali_data, self.finetune_vali_loader, criterion) # Use finetune data

            train_eval_loss, train_preds, train_trues, _ = self.vali(self.finetune_train_data, self.finetune_train_loader, criterion) # Use finetune data

            # --- Metric Processing (will use the new self.args.run_dir) ---
            train_reg_metrics = self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "latest")
            vali_reg_metrics = self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "latest")

            # --- Logging and History ---
            avg_grad_norm = np.mean(epoch_grad_norms)
            cost_time = time.time() - epoch_time
            epoch_time = time.time()
            remaining_time = cost_time * (epochs - epoch - 1)

            current_lr = model_optim.param_groups[0]['lr']
            history_train_loss.append(train_loss_avg); history_vali_loss.append(vali_loss); history_lr.append(current_lr)
            
            formatted_cost_time = format_duration(cost_time)
            formatted_eta = format_duration(remaining_time)

            log_msg = f"Finetune Epoch: {epoch + 1} /{epochs} | Train Loss: {train_loss_avg:.4f} | Vali Loss: {vali_loss:.4f}"
            log_msg += f" | Grad: {avg_grad_norm:.4f} | LR: {current_lr:.6f}"
            log_msg += f" | Time: {formatted_cost_time} | ETA: {formatted_eta}"
            
            if (epoch + 1) % self.args.vali_interval == 0:
                if train_reg_metrics: print(f"Finetune Train Metrics:\n{format_metrics(train_reg_metrics)}")
                if vali_reg_metrics: print(f"Finetune Validation Metrics:\n{format_metrics(vali_reg_metrics)}")

            # --- Early Stopping & Best Model Checkpoint Saving ---
            prev_best_loss = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, chechpoint_path)
            if vali_loss < prev_best_loss:
                self.logger.info(f"New best finetune validation loss: {vali_loss:.4f}. Model checkpoint saved.")

            current_feh_mae = vali_reg_metrics['FeH_mae']
            if current_feh_mae < best_feh_mae:
                best_feh_mae = current_feh_mae
                torch.save(self.model.state_dict(), chechpoint_path + '/' + 'best.pth')

                self.logger.info(f"New best finetune FeH MAE: {best_feh_mae:.4f}. Saving metrics for this epoch as 'best'...")
                self.calculate_and_save_all_metrics(train_preds, train_trues, "train", "best")
                self.calculate_and_save_all_metrics(vali_preds, vali_trues, "val", "best")

            if early_stopping.early_stop: break
            if scheduler is not None: scheduler.step() 

            save_history_plot(history_train_loss, history_vali_loss, history_lr, self.args.run_dir) # will save to finetune dir
            self.logger.info(log_msg)
        
        # The run_dir is intentionally NOT restored, so subsequent calls (like test)
        # will use the finetuning directory for checkpoints and results.
        self.logger.info(f"--- Finetuning Finished ---")
        return self.model


    def test(self):
        # 1. 加载最优模型
        checkpoint_path = os.path.join(self.args.run_dir, 'checkpoints', 'best.pth')
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading best model from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)
        else:
            self.logger.warning("No best model checkpoint found. Testing on the final model state.")

        # 2. 创建保存目录
        save_dir = os.path.join(self.args.run_dir, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        # 3. 在测试集上评估
        self.logger.info("--- Starting Final Test ---")
        
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
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)

        # --- Loop through each dataset (train, val, test) and evaluate ---
        dataset_splits = {
            'train': (self.train_data, self.train_loader),
            'val': (self.vali_data, self.vali_loader),
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
        return select_criterion(self.args.loss)        
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