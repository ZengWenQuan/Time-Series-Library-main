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

        # Import FeatureAdjuster locally for this specific logic
        from ..submodules.fusion_heads.fusion_modules import FeatureAdjuster

        # --- 1. Initialize Backbone (real or dummy) ---
        if 'backbone_config' in model_config:
            print("Initializing Backbone from config...")
            backbone_config = model_config['backbone_config']
            backbone_config.update(global_settings)
            self.backbone = BACKBONES[backbone_config['name']](backbone_config)
        else:
            print("No backbone configured. Creating a FeatureAdjuster dummy backbone.")
            raw_in_channels = global_settings.get('input_channels', 1)
            raw_in_len = global_settings.get('input_length', 4800)
            target_channels = global_settings.get('dummy_backbone_out_channels', 64)
            target_len = global_settings.get('dummy_backbone_out_len', 600)
            print(f"Dummy backbone will adjust input from ({raw_in_channels}ch, {raw_in_len}len) to ({target_channels}ch, {target_len}len).")
            self.backbone = FeatureAdjuster(target_channels, target_len)
            # Manually set output shape attributes for downstream modules
            self.backbone.output_channels = target_channels
            self.backbone.output_length = target_len

        # --- 2. Initialize Real Branches (if configured) ---
        branch_in_channels = self.backbone.output_channels
        branch_in_len = self.backbone.output_length
        print(f"Branches will receive input with: {branch_in_channels} channels, {branch_in_len} length.")

        self.global_branch = None
        if 'global_branch_config' in model_config:
            print(f"Initializing Global Branch from config...")
            g_conf = model_config['global_branch_config']
            g_conf.update(global_settings)
            g_conf['in_channels'] = branch_in_channels
            g_conf['in_len'] = branch_in_len
            self.global_branch = GLOBAL_BRANCH_REGISTRY[g_conf['name']](g_conf)

        self.local_branch = None
        if 'local_branch_config' in model_config:
            print(f"Initializing Local Branch from config...")
            l_conf = model_config['local_branch_config']
            l_conf.update(global_settings)
            l_conf['in_channels'] = branch_in_channels
            l_conf['in_len'] = branch_in_len
            self.local_branch = LOCAL_BRANCH_REGISTRY[l_conf['name']](l_conf)

        # --- 3. Create FeatureAdjuster Placeholders for Missing Branches ---
        if self.global_branch is None and self.local_branch is None:
            raise ValueError("At least one branch (global or local) must be configured.")

        elif self.global_branch is None:
            print("Global branch is missing. Creating a FeatureAdjuster placeholder to match local_branch output.")
            target_channels = self.local_branch.output_channels
            target_len = self.local_branch.output_length
            self.global_branch = FeatureAdjuster(target_channels, target_len)
            # Manually set output shape attributes
            self.global_branch.output_channels = target_channels
            self.global_branch.output_length = target_len

        elif self.local_branch is None:
            print("Local branch is missing. Creating a FeatureAdjuster placeholder to match global_branch output.")
            target_channels = self.global_branch.output_channels
            target_len = self.global_branch.output_length
            self.local_branch = FeatureAdjuster(target_channels, target_len)
            # Manually set output shape attributes
            self.local_branch.output_channels = target_channels
            self.local_branch.output_length = target_len

        # --- 4. Initialize Fusion and Head (Unconditionally) ---
        # After step 3, both branches always exist and have compatible shapes.
        # The fusion module is now always expected to be configured.
        print("Initializing Fusion Module...")
        if 'fusion_config' not in model_config or 'fusion_name' not in model_config:
            raise ValueError("A 'fusion_config' and 'fusion_name' must be provided.")
        
        fusion_config = model_config['fusion_config']
        fusion_config.update(global_settings)
        fusion_config['channels_cont'] = self.global_branch.output_channels
        fusion_config['len_cont'] = self.global_branch.output_length
        fusion_config['channels_norm'] = self.local_branch.output_channels
        fusion_config['len_norm'] = self.local_branch.output_length
        
        self.fusion = FUSION_REGISTRY[model_config['fusion_name']](fusion_config)
        head_input_channels = self.fusion.output_channels
        head_input_length = self.fusion.output_length

        # --- 5. Initialize Prediction Head ---
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
        macs, params = profile(self.global_branch, inputs=(branch_input,), verbose=False)
        stats['global_branch'] = {'flops': macs * 2, 'params': params}
        features_global = self.global_branch(branch_input)
        
        macs, params = profile(self.local_branch, inputs=(branch_input,), verbose=False)
        stats['local_branch'] = {'flops': macs * 2, 'params': params}
        features_local = self.local_branch(branch_input)

        # 3. Profile Fusion
        macs, params = profile(self.fusion, inputs=(features_local, features_global), verbose=False)
        stats['fusion'] = {'flops': macs * 2, 'params': params}
        head_input = self.fusion(features_local, features_global)

        # 4. Profile Head
        macs, params = profile(self.prediction_head, inputs=(head_input,), verbose=False)
        stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats

    def forward(self, x, return_features=False):
        if x.dim() == 2: x = x.unsqueeze(1)

        # 1. Backbone Pass
        branch_input = self.backbone(x)

        # 2. Branch Pass (real or placeholder)
        features_global = self.global_branch(branch_input)
        features_local = self.local_branch(branch_input)

        # 3. Fusion (Unconditional)
        head_input = self.fusion(features_local, features_global)
        
        if return_features:
            return head_input, features_local, features_global

        # 4. Prediction Head Pass
        return self.prediction_head(head_input)
