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
            #print(f"Backbone output - channels: {self.backbone.output_channels}, length: {self.backbone.output_length}")

        # --- Dynamically build the model based on config ---
        head_input_channels = 0
        head_input_length = 0

        # 2. Initialize Global Branch (if configured)
        if 'global_branch_config' in model_config:
            print("Initializing Global Branch...")
            global_branch_config = model_config['global_branch_config']
            global_branch_config.update(global_settings)

            # 如果有backbone，更新global_branch的输入参数
            if self.backbone:
                global_branch_config['in_channels'] = self.backbone.output_channels
                global_branch_config['in_len'] = self.backbone.output_length

            GlobalBranchClass = GLOBAL_BRANCH_REGISTRY[global_branch_config['name']]
            self.global_branch = GlobalBranchClass(global_branch_config)
            print(f"Global Branch output - channels: {self.global_branch.output_channels}, length: {self.global_branch.output_length}")

        # 3. Initialize Local Branch (if configured)
        if 'local_branch_config' in model_config :
            print("Initializing Local Branch...")
            local_branch_config = model_config['local_branch_config']
            local_branch_config.update(global_settings)

            # 如果有backbone，更新local_branch的输入参数
            if self.backbone: #如果有backbone，输入通道数和长度使用backbone的输出，否则使用全局的input_channels,input_len
                local_branch_config['in_channels'] = self.backbone.output_channels
                local_branch_config['in_len'] = self.backbone.output_length
                print(self.backbone.output_channels,'backbone通道数')

            LocalBranchClass = LOCAL_BRANCH_REGISTRY[local_branch_config['name']]
            self.local_branch = LocalBranchClass(local_branch_config)
            print(f"Local Branch output - channels: {self.local_branch.output_channels}, length: {self.local_branch.output_length}")

        # 4. Initialize Fusion module (if configured)
        if 'fusion_config' in model_config and 'fusion_name' in model_config:
            print("Initializing Fusion Module (allowing single branch)...")
            fusion_config = model_config['fusion_config']
            fusion_config.update(global_settings)
            FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
            
            self.fusion = FusionClass(fusion_config)
            head_input_channels=self.fusion.output_channels
            print(f"Fusion output - channels: {self.fusion.output_channels}, length: {self.fusion.output_length}")

        if head_input_channels == 0:
            raise ValueError("No branches were configured. The model has no inputs for the prediction head.")

        # 5. Initialize Prediction Head (always required)
        if 'head_config' in model_config :
            print("Initializing Prediction Head...")
            head_config = model_config['head_config']
            head_config.update(global_settings)
            HeadClass = HEAD_REGISTRY[head_config['name']]
            head_config['input_channels'] =  self.fusion.output_channels
            head_config['input_length'] = self.fusion.output_length
            head_config['targets'] = self.targets
            print(f"Head input - channels: {head_input_channels}, length: {head_input_length}")
            self.prediction_head = HeadClass(head_config)
            print(f"targets: {len(self.targets)}")
        else:
            raise ValueError("A 'head_config' must be provided in the model configuration.")


    def profile_model(self, sample_batch):
        """
        Calculates and returns the FLOPs and parameters for each submodule.
        This method is robust to missing modules for ablation studies.
        """
        from thop import profile
        stats = {}
        x = sample_batch

        # --- Input Preprocessing for Single-Channel Data ---
        if x.dim() == 2:  # Handle (B, L) format
            x = x.unsqueeze(1)  # -> (B, 1, L)

        # --- Backbone Forward Pass ---
        if self.backbone:
            macs, params = profile(self.backbone, inputs=(x,), verbose=False)
            stats['backbone'] = {'flops': macs * 2, 'params': params}
            backbone_output = self.backbone(x)
        else:
            backbone_output = x

        if self.global_branch:
            macs, params = profile(self.global_branch, inputs=(backbone_output,), verbose=False)
            stats['global_branch'] = {'flops': macs * 2, 'params': params}
            features_global = self.global_branch(backbone_output)
        else :        
            features_global = backbone_output

        if self.local_branch:
            macs, params = profile(self.local_branch, inputs=(backbone_output,), verbose=False)
            stats['local_branch'] = {'flops': macs * 2, 'params': params}
            features_local = self.local_branch(backbone_output)
        else :
            features_local = backbone_output

        if self.fusion:
            macs, params = profile(self.fusion, inputs=(features_local, features_global), verbose=False)
            stats['fusion'] = {'flops': macs * 2, 'params': params}
            
            head_input = self.fusion(features_local, features_global)

        if self.prediction_head :
            macs, params = profile(self.prediction_head, inputs=(head_input,), verbose=False)
            stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats
                

    def forward(self, x):
        # --- Input Preprocessing for Single-Channel Data ---
        if x.dim() == 2:  # Handle (B, L) format
            x = x.unsqueeze(1)  # -> (B, 1, L)

        # 2. 通过骨干网络
        if self.backbone:
            backbone_output = self.backbone(x)
        else:
            backbone_output = x

        # 3. 正确地获取分支输出，如果分支不存在则为 None
        features_global = self.global_branch(backbone_output) if self.global_branch is not None else None
        features_local = self.local_branch(backbone_output) if self.local_branch is not None else None

        # 4. 融合模块必须能处理 None 输入
        # (例如，如果一个分支输出为 None，它应该只返回另一个分支的输出)
        head_input = self.fusion(features_local, features_global)

        # 5. 如果融合后无输出，则报错
        if head_input is None:
            raise ValueError("Fusion module returned None. Check branch and fusion logic.")

        return self.prediction_head(head_input)