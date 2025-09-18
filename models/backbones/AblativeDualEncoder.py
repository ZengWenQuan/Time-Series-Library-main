# /home/irving/workspace/Time-Series-Library-main/models/backbones/AblativeDualEncoder.py

import torch
import torch.nn as nn
from models.registries import register_backbone, BLOCKS
from omegaconf import DictConfig

@register_backbone
class AblativeDualEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.initial_extractor = BLOCKS[cfg.initial_extractor.name](cfg.initial_extractor)
        
        self.branches = nn.ModuleDict()
        if 'branches' in cfg and cfg.branches is not None:
            for branch_cfg in cfg.branches:
                if branch_cfg: # 允许通过将分支设为 null 来进行消融
                    self.branches[branch_cfg.name] = BLOCKS[branch_cfg.name](branch_cfg)

    def forward(self, x):
        base_features = self.initial_extractor(x)
        
        if not self.branches:
            # 如果没有配置任何分支，直接返回初始特征
            return base_features

        branch_outputs = [branch(base_features) for branch in self.branches.values()]
        output = torch.cat(branch_outputs, dim=1)
        return output