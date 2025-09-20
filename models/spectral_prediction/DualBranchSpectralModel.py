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
    """
    def __init__(self, configs):
        super(DualBranchSpectralModel, self).__init__()
        #self.task_name = configs.task_name
        self.targets = configs['targets']
        model_config=configs

        global_settings = model_config.get('global_settings', {})

        # --- Initialize all modules to None ---
        self.backbone = None
        self.global_branch = None
        self.local_branch = None
        self.fusion = None
        self.prediction_head = None

        # --- 1. Initialize Backbone (if configured) ---
        if 'backbone_config' in model_config:
            print("Initializing Backbone...")
            backbone_config = model_config['backbone_config']
            backbone_config.update(global_settings)
            BackboneClass = BACKBONES[backbone_config['name']]
            self.backbone = BackboneClass(backbone_config)

        has_global = 'global_branch_config' in model_config
        has_local = 'local_branch_config' in model_config

        # --- Conditional Channel Allocation ---
        if self.backbone:
            backbone_channels = self.backbone.output_channels
            backbone_length = self.backbone.output_length
            # If backbone exists and both branches are present, split channels
            if has_global and has_local:
                print(f"Splitting backbone output ({backbone_channels} channels) for dual branches.")
                global_in_channels = backbone_channels // 2
                local_in_channels = backbone_channels - global_in_channels
            else: # Single branch case with backbone
                global_in_channels = local_in_channels = backbone_channels
        else:
            # No backbone (ablation study): both branches get the raw input
            print("No backbone. Branches will receive full raw input.")
            input_channels = global_settings.get('input_channels', 1)
            backbone_length = global_settings.get('input_len', 5000)
            global_in_channels = local_in_channels = input_channels

        if has_global:
            print(f"Initializing Global Branch with {global_in_channels} input channels...")
            g_conf = model_config['global_branch_config']
            g_conf.update(global_settings)
            g_conf['in_channels'] = global_in_channels
            g_conf['in_len'] = backbone_length
            self.global_branch = GLOBAL_BRANCH_REGISTRY[g_conf['name']](g_conf)

        if has_local:
            print(f"Initializing Local Branch with {local_in_channels} input channels...")
            l_conf = model_config['local_branch_config']
            l_conf.update(global_settings)
            l_conf['in_channels'] = local_in_channels
            l_conf['in_len'] = backbone_length
            self.local_branch = LOCAL_BRANCH_REGISTRY[l_conf['name']](l_conf)

        if 'fusion_config' in model_config and 'fusion_name' in model_config:
            print("Initializing Fusion Module...")
            fusion_config = model_config['fusion_config']
            fusion_config.update(global_settings)
            self.fusion = FUSION_REGISTRY[model_config['fusion_name']](fusion_config)
        else:
            raise ValueError("A 'fusion_config' and 'fusion_name' must be provided.")

        if 'head_config' in model_config:
            print("Initializing Prediction Head...")
            head_config = model_config['head_config']
            head_config.update(global_settings)
            head_config['input_channels'] = self.fusion.output_channels
            head_config['input_length'] = self.fusion.output_length
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

        # 1. Profile Backbone
        branch_input = self.backbone(x) if self.backbone else x
        if self.backbone:
            macs, params = profile(self.backbone, inputs=(x,), verbose=False)
            stats['backbone'] = {'flops': macs * 2, 'params': params}

        # 2. Profile Branches with correct conditional logic
        features_global, features_local = None, None

        # Case 1: Backbone exists and both branches exist (split channels)
        if self.backbone and self.global_branch and self.local_branch:
            mid_point = branch_input.shape[1] // 2
            global_input = branch_input[:, :mid_point, :]
            local_input = branch_input[:, mid_point:, :]

            macs, params = profile(self.global_branch, inputs=(global_input,), verbose=False)
            stats['global_branch'] = {'flops': macs * 2, 'params': params}
            features_global = self.global_branch(global_input)

            macs, params = profile(self.local_branch, inputs=(local_input,), verbose=False)
            stats['local_branch'] = {'flops': macs * 2, 'params': params}
            features_local = self.local_branch(local_input)
        else:
            # Case 2: No backbone or single branch (shared input)
            if self.global_branch:
                macs, params = profile(self.global_branch, inputs=(branch_input,), verbose=False)
                stats['global_branch'] = {'flops': macs * 2, 'params': params}
                features_global = self.global_branch(branch_input)
            if self.local_branch:
                macs, params = profile(self.local_branch, inputs=(branch_input,), verbose=False)
                stats['local_branch'] = {'flops': macs * 2, 'params': params}
                features_local = self.local_branch(branch_input)

        # 3. Profile Fusion
        if self.fusion is None: raise ValueError("Fusion module is not configured.")
        macs, params = profile(self.fusion, inputs=(features_local, features_global), verbose=False)
        stats['fusion'] = {'flops': macs * 2, 'params': params}
        head_input = self.fusion(features_local, features_global)

        # 4. Profile Head
        if self.prediction_head is None: raise ValueError("Prediction head is not configured.")
        macs, params = profile(self.prediction_head, inputs=(head_input,), verbose=False)
        stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats

    def forward(self, x, return_features=False):
        if x.dim() == 2: x = x.unsqueeze(1)

        branch_input = self.backbone(x) if self.backbone else x

        features_global, features_local = None, None

        # If backbone exists and both branches exist, split the output
        if self.backbone and self.global_branch and self.local_branch:
            mid_point = branch_input.shape[1] // 2
            features_global = self.global_branch(branch_input[:, :mid_point, :])
            features_local = self.local_branch(branch_input[:, mid_point:, :])
        else:
            # Otherwise, both branches receive the same full input (either raw x or full backbone output)
            if self.global_branch:
                features_global = self.global_branch(branch_input)
            if self.local_branch:
                features_local = self.local_branch(branch_input)

        if self.fusion is None: raise ValueError("Fusion module is not configured.")
        head_input = self.fusion(features_local, features_global)
        if head_input is None: raise ValueError("Fusion module returned None.")

        if return_features:
            return head_input, features_local, features_global

        return self.prediction_head(head_input)
