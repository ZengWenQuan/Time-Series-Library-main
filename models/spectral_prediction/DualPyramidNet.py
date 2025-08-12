
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
from exp.exp_basic import register_model
from models.submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY

@register_model('DualPyramidNet')
class DualPyramidNet(nn.Module):
    def __init__(self, configs):
        super(DualPyramidNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size

        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # --- 使用注册器动态构建分支 ---
        BranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['branch_name']]
        branch_config = model_config['branch_config']
        
        self.continuum_extractor = BranchClass(branch_config)
        self.normalized_extractor = BranchClass(branch_config)
        
        ffn_input_dim = self.continuum_extractor.output_dim + self.normalized_extractor.output_dim
        
        # --- FFN 预测头 ---
        fc_layers = []
        current_dim = ffn_input_dim
        for hidden_dim in model_config['prediction_head']['fc_hidden_dims']:
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            if model_config['branch_config']['use_batch_norm']: # 遵循分支的BN设置
                fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(model_config['prediction_head']['dropout']))
            current_dim = hidden_dim
        
        fc_layers.append(nn.Linear(current_dim, self.label_size))
        self.ffn = nn.Sequential(*fc_layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear): init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d): init.constant_(m.weight, 1); init.constant_(m.bias, 0)

    def forward(self, x, x_normalized=None):
        # 回归任务只使用一个输入
        if self.task_name == 'regression':
            features = self.normalized_extractor(x)
            return self.ffn(features)
        
        # 光谱预测任务使用两个输入
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            
            continuum_features = self.continuum_extractor(x_continuum)
            normalized_features = self.normalized_extractor(x_normalized)
            combined_features = torch.cat([continuum_features, normalized_features], dim=1)
            return self.ffn(combined_features)
        
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")
