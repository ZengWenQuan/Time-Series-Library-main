import os
import torch
import torch.nn as nn
from utils.losses import RegressionFocalLoss
import logging

# --- Model Registration Mechanism ---
MODEL_REGISTRY = {}

def register_model(name):
    """A decorator to register a new model class."""
    def decorator(cls):
        if cls.__name__ in MODEL_REGISTRY:
            raise ValueError(f'模型{cls.__name__} 已经存在了！不能重复注册')
        MODEL_REGISTRY[cls.__name__] = cls
        return cls
    return decorator
# --- End of Mechanism ---

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_class = MODEL_REGISTRY.get(self.args.model)
        if model_class is None:
            raise ValueError(f"Model '{self.args.model}' is not registered. "
                             f"Available models: {list(MODEL_REGISTRY.keys())}")
        
        print(f"Building model: {self.args.model}")
        model = model_class(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        info_model=self.args.run_dir+'/model.txt'
        with open(info_model,'w') as f:
            # 写入模型结构
            f.write("模型结构:\n")
            f.write(f"{model}\n\n")
            
            # 写入每层参数数量
            f.write("每层参数数量:\n")
            sum_param=0
            for name, param in model.named_parameters():                
                f.write(f"  {name}: {param.numel():,} 参数\n")
                sum_param+=param.numel()
            f.write(f'总参数量：{sum_param}')
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
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _select_criterion(self):
        if not hasattr(self.args, 'loss') or self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAE':
            return nn.L1Loss()
        elif self.args.loss == 'SmoothL1':
            return nn.SmoothL1Loss()
        elif self.args.loss == 'Huber':
            return nn.HuberLoss(delta=1.0)
        elif self.args.loss == 'LogCosh':
            def logcosh_loss(pred, target):
                return torch.mean(torch.log(torch.cosh(pred - target)))
            return logcosh_loss
        elif self.args.loss == 'FocalMSE':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mse')
        elif self.args.loss == 'FocalMAE':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mae')
        elif self.args.loss == 'FocalSmoothL1':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='smooth_l1')
        else:
            print(f"警告: 未知的损失函数 '{self.args.loss}'，使用默认的MSE损失")
            return nn.MSELoss()
        
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