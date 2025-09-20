import torch
import torch.nn as nn
import yaml
from ..registries import (
    register_model,
    BACKBONES,
    GLOBAL_BRANCH_REGISTRY,
    LOCAL_BRANCH_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY
)

@register_model
class DualBranchSpectralModel(nn.Module):
    """
    双分支光谱预测模型。
    该模型具有明确的backbone结构，包含两个分支和一个预测头。
    此版本经过重构，以支持灵活的消融实验，无论主干、全局或局部分支是否存在，都能正常工作。
    当主干网络(backbone)未在配置中指定时，会自动创建一个虚拟主干(dummy backbone)——一个简单的卷积层，
    以确保下游分支接收到维度正确的输入。
    """
    def __init__(self, configs):
        super(DualBranchSpectralModel, self).__init__()
        self.targets = configs['targets']
        model_config = configs
        global_settings = model_config.get('global_settings', {})

        # --- 1. Initialize all modules to None for flexible ablation ---
        self.backbone = None
        self.global_branch = None
        self.local_branch = None
        self.fusion = None
        self.prediction_head = None

        # --- 2. Initialize Backbone (real or dummy) ---
        if 'backbone_config' in model_config:
            print("Initializing Backbone from config...")
            backbone_config = model_config['backbone_config']
            backbone_config.update(global_settings)
            self.backbone = BACKBONES[backbone_config['name']](backbone_config)
        else:
            # No backbone configured, create a "dummy" projection backbone
            print("No backbone configured. Creating a dummy projection backbone (Conv1d).")
            raw_in_channels = global_settings.get('input_channels', 1)
            input_len = global_settings.get('input_length', 5000)
            
            # Use a configurable projected channel size, default to 16
            projected_channels = global_settings.get('projected_channels', 16)
            
            kernel_size = 3
            stride = 2
            # Use padding to ensure the formula is clean and reduces length by approx. stride factor
            padding = 1 
            
            self.backbone = nn.Conv1d(
                in_channels=raw_in_channels,
                out_channels=projected_channels,
                kernel_size=3,
                stride=2,
                padding=padding
            )
            
            # Manually calculate and attach output shape attributes to the dummy backbone
            self.backbone.output_channels = projected_channels
            
            # L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
            output_length = (input_len + 2 * padding - kernel_size) // stride + 1
            self.backbone.output_length = output_length
            
            print(f"Dummy backbone projects from {raw_in_channels}ch to {projected_channels}ch. Length from {input_len} to {output_length}.")

        # --- 3. Determine input dimensions for the branches ---
        # This part now works universally because self.backbone always exists (either real or dummy)
        branch_in_channels = self.backbone.output_channels
        branch_in_len = self.backbone.output_length
        print(f"Branches will receive input with: {branch_in_channels} channels, {branch_in_len} length.")

        # --- 4. Initialize Branches (if configured) ---
        if 'global_branch_config' in model_config:
            print(f"Initializing Global Branch...")
            g_conf = model_config['global_branch_config']
            g_conf.update(global_settings)
            g_conf['in_channels'] = branch_in_channels
            g_conf['in_len'] = branch_in_len
            self.global_branch = GLOBAL_BRANCH_REGISTRY[g_conf['name']](g_conf)

        if 'local_branch_config' in model_config:
            print(f"Initializing Local Branch...")
            l_conf = model_config['local_branch_config']
            l_conf.update(global_settings)
            l_conf['in_channels'] = branch_in_channels
            l_conf['in_len'] = branch_in_len
            self.local_branch = LOCAL_BRANCH_REGISTRY[l_conf['name']](l_conf)

        # --- 5. Determine Head Input and Initialize Fusion (if needed) ---
        head_input_channels = 0
        head_input_length = 0

        if self.global_branch and self.local_branch:
            # Both branches exist, fusion is required
            print("Both branches found. Initializing Fusion Module...")
            if 'fusion_config' not in model_config or 'fusion_name' not in model_config:
                raise ValueError("With two branches, 'fusion_config' and 'fusion_name' must be provided.")
            
            fusion_config = model_config['fusion_config']
            fusion_config.update(global_settings)
            # Provide branch output info to fusion module for its own setup
            if hasattr(self.global_branch, 'output_channels'):
                fusion_config['channels_cont'] = self.global_branch.output_channels
                fusion_config['len_cont'] = self.global_branch.output_length
            if hasattr(self.local_branch, 'output_channels'):
                fusion_config['channels_norm'] = self.local_branch.output_channels
                fusion_config['len_norm'] = self.local_branch.output_length
            
            self.fusion = FUSION_REGISTRY[model_config['fusion_name']](fusion_config)
            head_input_channels = self.fusion.output_channels
            head_input_length = self.fusion.output_length
        elif self.global_branch:
            # Only global branch exists, bypass fusion
            print("Only global branch found. Output will be passed directly to head.")
            head_input_channels = self.global_branch.output_channels
            head_input_length = self.global_branch.output_length
        elif self.local_branch:
            # Only local branch exists, bypass fusion
            print("Only local branch found. Output will be passed directly to head.")
            head_input_channels = self.local_branch.output_channels
            head_input_length = self.local_branch.output_length
        else:
            raise ValueError("Invalid model configuration: At least one branch (global or local) must be defined.")

        # --- 6. Initialize Prediction Head ---
        if 'head_config' in model_config:
            print(f"Initializing Prediction Head with {head_input_channels} input channels and {head_input_length} length...")
            head_config = model_config['head_config']
            head_config.update(global_settings)
            head_config['input_channels'] = head_input_channels
            head_config['input_length'] = head_input_length
            head_config['targets'] = self.targets
            self.prediction_head = HEAD_REGISTRY[head_config['name']](head_config)
        else:
            raise ValueError("A 'head_config' must be provided.")

    def profile_model(self, sample_batch):
        """
        Calculates and returns the FLOPs and parameters for each submodule.
        This method is robust to missing modules for ablation studies.
        """
        from thop import profile
        stats = {}
        x = sample_batch
        if x.dim() == 2: x = x.unsqueeze(1)

        # 1. Profile Backbone (real or dummy)
        macs, params = profile(self.backbone, inputs=(x,), verbose=False)
        stats['backbone'] = {'flops': macs * 2, 'params': params}
        branch_input = self.backbone(x)

        # 2. Profile Branches
        features_global, features_local = None, None
        if self.global_branch:
            macs, params = profile(self.global_branch, inputs=(branch_input,), verbose=False)
            stats['global_branch'] = {'flops': macs * 2, 'params': params}
            features_global = self.global_branch(branch_input)
        if self.local_branch:
            macs, params = profile(self.local_branch, inputs=(branch_input,), verbose=False)
            stats['local_branch'] = {'flops': macs * 2, 'params': params}
            features_local = self.local_branch(branch_input)

        # 3. Profile Fusion or handle single-branch passthrough
        head_input = None
        if self.fusion and features_global is not None and features_local is not None:
            macs, params = profile(self.fusion, inputs=(features_local, features_global), verbose=False)
            stats['fusion'] = {'flops': macs * 2, 'params': params}
            head_input = self.fusion(features_local, features_global)
        elif features_global is not None:
            head_input = features_global
        elif features_local is not None:
            head_input = features_local
        
        if head_input is None:
            raise RuntimeError("Profiling failed: No features were generated to profile the head.")

        # 4. Profile Head
        macs, params = profile(self.prediction_head, inputs=(head_input,), verbose=False)
        stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats

    def forward(self, x, return_features=False):
        if x.dim() == 2: x = x.unsqueeze(1)

        # 1. Backbone Pass (real or dummy)
        branch_input = self.backbone(x)

        # 2. Branch Pass (if they exist)
        features_global = self.global_branch(branch_input) if self.global_branch else None
        features_local = self.local_branch(branch_input) if self.local_branch else None

        # 3. Fusion or Passthrough Logic
        head_input = None
        if self.fusion and features_global is not None and features_local is not None:
            # Both branches ran, use fusion module
            head_input = self.fusion(features_local, features_global)
        elif features_global is not None:
            # Only global branch ran
            head_input = features_global
        elif features_local is not None:
            # Only local branch ran
            head_input = features_local
        
        if head_input is None:
            raise RuntimeError("Invalid forward pass: No features were generated by any branch.")

        if return_features:
            return head_input, features_local, features_global

        # 4. Prediction Head Pass
        return self.prediction_head(head_input)
