import torch
import torch.nn as nn
import yaml
from ..registries import (
    register_model,
    CONTINUUM_BRANCH_REGISTRY,
    NORMALIZED_BRANCH_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY
)

@register_model
class GenericSpectralModel(nn.Module):
    """
    通用的光谱预测主模型。
    该模型的设计是完全配置驱动的，其__init__和forward逻辑模仿了项目中的标准实现，
    使其能够灵活处理spectral_prediction和regression等不同任务。
    """
    def __init__(self, configs):
        super(GenericSpectralModel, self).__init__()
        self.task_name = configs['task_name']
        self.targets = configs['targets']

        # with open(configs.model_conf, 'r') as f:
        #     self.model_config = yaml.safe_load(f)
        self.model_config = configs

        global_settings = self.model_config.get('global_settings', {})

        # Initialize all modules to None
        self.global_branch = None
        self.local_branch = None
        self.fusion = None
        self.prediction_head = None

        # --- Dynamically build the model based on config ---
        head_input_channels = 0
        head_input_length = 0

        # 1. Initialize Continuum Branch (if configured)
        if 'global_branch_config' in self.model_config and 'global_branch_name' in self.model_config:
            print("Initializing global Branch...")
            global_branch_config = self.model_config['global_branch_config']
            global_branch_config.update(global_settings)
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[self.model_config['global_branch_name']]
            self.global_branch = ContBranchClass(global_branch_config)

        # 2. Initialize Normalized Branch (if configured)
        if 'local_branch_config' in self.model_config and 'local_branch_name' in self.model_config:
            print("Initializing Normalized Branch...")
            local_branch_config = self.model_config['local_branch_config']
            local_branch_config.update(global_settings)
            NormBranchClass = NORMALIZED_BRANCH_REGISTRY[self.model_config['local_branch_name']]
            self.local_branch = NormBranchClass(local_branch_config)

        # 3. Initialize Fusion module (if configured)
        if 'fusion_config' in self.model_config and 'fusion_name' in self.model_config:
            print("Initializing Fusion Module (allowing single branch)...")
            fusion_config = self.model_config['fusion_config']
            fusion_config.update(global_settings)
            FusionClass = FUSION_REGISTRY[self.model_config['fusion_name']]

            # Conditionally provide branch info to fusion config
            if self.local_branch:
                fusion_config['channels_norm'] = self.local_branch.output_channels
                fusion_config['len_norm'] = self.local_branch.output_length
            if self.global_branch:
                fusion_config['channels_cont'] = self.global_branch.output_channels
                fusion_config['len_cont'] = self.global_branch.output_length

            self.fusion = FusionClass(fusion_config)
            
            # Determine head input based on what will be produced after fusion/bypass
            if self.global_branch and self.local_branch:
                head_input_channels = self.fusion.output_channels
                head_input_length = self.fusion.output_length
            elif self.global_branch:
                head_input_channels = self.global_branch.output_channels
                head_input_length = self.global_branch.output_length
            elif self.local_branch:
                head_input_channels = self.local_branch.output_channels
                head_input_length = self.local_branch.output_length
        else:
            # No fusion module, determine head input from available branches by concatenation
            print("No Fusion module configured. Head input will be concatenation of available branch outputs.")
            if self.global_branch:
                head_input_channels += self.global_branch.output_channels
                head_input_length = self.global_branch.output_length # Assume lengths are compatible
            if self.local_branch:
                head_input_channels += self.local_branch.output_channels
                head_input_length = self.local_branch.output_length # Overwrite or check for consistency

        if head_input_channels == 0:
            raise ValueError("No branches were configured. The model has no inputs for the prediction head.")

        # 4. Initialize Prediction Head (always required)
        if 'head_config' in self.model_config and 'head_name' in self.model_config:
            print("Initializing Prediction Head...")
            head_config = self.model_config['head_config']
            head_config.update(global_settings)
            HeadClass = HEAD_REGISTRY[self.model_config['head_name']]
            head_config['head_input_channels'] = head_input_channels
            head_config['head_input_length'] = head_input_length
            head_config['targets'] = self.targets
            self.prediction_head = HeadClass(head_config)
        else:
            raise ValueError("A 'head_config' must be provided in the model configuration.")


    def profile_model(self, sample_batch):
        """
        Calculates and returns the FLOPs and parameters for each submodule.
        This method is robust to missing modules for ablation studies.
        """
        from thop import profile
        stats = {}
        device = sample_batch.device

        # --- Input Handling ---
        # x_continuum, x_normalized = None, None
        # if self.task_name == 'regression':
        #     x_continuum = x_normalized = sample_batch
        # elif self.task_name == 'spectral_prediction':
        #     if sample_batch.dim() == 3 and sample_batch.shape[-1] == 2:
        #         x_continuum = sample_batch[:, :, 0].unsqueeze(1)
        #         x_normalized = sample_batch[:, :, 1].unsqueeze(1)
        #     elif sample_batch.dim() == 2 and sample_batch.shape[-1] == 2:
        #         x_continuum = sample_batch[:, 0].unsqueeze(1)
        #         x_normalized = sample_batch[:, 1].unsqueeze(1)
        sample_batch = sample_batch.unsqueeze(1)
        x_continuum=sample_batch
        x_normalized=sample_batch
        # --- Profile available modules ---
        features_cont, features_norm = None, None
        head_input = None

        if self.global_branch and x_continuum is not None:
            macs, params = profile(self.global_branch, inputs=(x_continuum,), verbose=False)
            stats['global_branch'] = {'flops': macs * 2, 'params': params}
            features_cont = self.global_branch(x_continuum)

        if self.local_branch and x_normalized is not None:
            macs, params = profile(self.local_branch, inputs=(x_normalized,), verbose=False)
            stats['local_branch'] = {'flops': macs * 2, 'params': params}
            features_norm = self.local_branch(x_normalized)

        if self.fusion:
            if features_norm is None or features_cont is None:
                raise ValueError("Profiling fusion module requires both branches to be active.")
            macs, params = profile(self.fusion, inputs=(features_norm, features_cont), verbose=False)
            stats['fusion'] = {'flops': macs * 2, 'params': params}
            head_input = self.fusion(features_norm, features_cont)
        else:
            existing_features = [f for f in [features_cont, features_norm] if f is not None]
            if len(existing_features) > 1:
                head_input = torch.cat(existing_features, dim=1)
            elif existing_features:
                head_input = existing_features[0]

        if self.prediction_head and head_input is not None:
            macs, params = profile(self.prediction_head, inputs=(head_input,), verbose=False)
            stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats
                

    def forward(self, x):
        # --- Input Handling based on task ---
        # x_continuum, x_normalized = None, None
        # if self.task_name == 'regression':
        #     # For regression, the same input 'x' is used for all present branches
        #     x_continuum = x if self.global_branch else None
        #     x_normalized = x if self.local_branch else None
        # elif self.task_name == 'spectral_prediction':
        #     # For spectral prediction, input 'x' is expected to have two channels
        #     if x.dim() == 3 and x.shape[-1] == 2:
        #         x_continuum = x[:, :, 0].unsqueeze(1) if self.global_branch else None
        #         x_normalized = x[:, :, 1].unsqueeze(1) if self.local_branch else None
        #     else:
        #         raise ValueError(f"Unexpected input shape for spectral_prediction: {x.shape}. Expected (batch, seq_len, 2).")
        # else:
        #     raise ValueError(f"Unknown task_name: {self.task_name}")

        # # --- Branch Forward Pass ---
        x = x.unsqueeze(1)
        x_continuum=x
        x_normalized=x
        features_cont = self.global_branch(x_continuum) if self.global_branch and x_continuum is not None else None
        features_norm = self.local_branch(x_normalized) if self.local_branch and x_normalized is not None else None

        # --- Fusion or Bypass/Concatenation ---
        head_input = None
        existing_features = [f for f in [features_cont, features_norm] if f is not None]

        if not existing_features:
            raise ValueError("No branches were configured or processed to provide features to the head.")

        if self.fusion:
            if len(existing_features) == 2:
                # Both branches are active, perform fusion. Order (norm, cont) is important for many modules.
                head_input = self.fusion(features_norm, features_cont)
            elif len(existing_features) == 1:
                # Only one branch is active, bypass the fusion module.
                head_input = existing_features[0]
        else:
            # No fusion module configured, default to concatenation.
            if len(existing_features) > 1:
                head_input = torch.cat(existing_features, dim=1)
            else:
                head_input = existing_features[0]

        # --- Head Forward Pass ---
        if self.prediction_head is None:
            raise ValueError("Prediction head is not initialized.")
            
        return self.prediction_head(head_input)

    