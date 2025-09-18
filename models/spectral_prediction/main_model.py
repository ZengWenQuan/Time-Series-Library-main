import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..registries import register_model
from .GenericSpectralModel import GenericSpectralModel

@register_model
class DualBranchMoENet(GenericSpectralModel):
    pass

@register_model
class DualSpectralNet(GenericSpectralModel):
    pass

@register_model
class LargeKernelConvNet(GenericSpectralModel):
    pass

@register_model
class CustomFusionNet(GenericSpectralModel):
    pass

@register_model
class FlexibleFusionNet(GenericSpectralModel):
    """
    A flexible fusion network that inherits all functionality directly from
    GenericSpectralModel. It is registered under its own name to allow for
    separate configuration and tracking, while reusing the generic model logic.
    """
    pass

@register_model
class MPFiLM(GenericSpectralModel):
    """
    MPFiLM (Micn-PatchTST-FiLM) model.
    Inherits all functionality from GenericSpectralModel.
    - Continuum Branch: MicnBranch
    - Normalized Branch: PatchTstBranch
    - Fusion: FilmFusion
    - Head: GlobalProjectionHead
    """
    pass

@register_model
class HybridFormer(GenericSpectralModel):
    """
    HybridFormer (CNN-Transformer + PatchTST) model.
    Inherits all functionality from GenericSpectralModel.
    - Continuum Branch: CnnTransformerBranch
    - Normalized Branch: PatchTstBranch
    - Fusion: GruFusion
    - Head: DecoderHead
    """
    pass

@register_model
class AMFN(GenericSpectralModel):
    """
    AMFN (Attention Multi-scale Fusion Network) model.
    Inherits all functionality from GenericSpectralModel.
    - Continuum Branch: GlobalAttentionBranch
    - Normalized Branch: MultiScaleConvBranch
    - Fusion: add
    - Head: DecoderHead
    """
    pass

# 注意：DualBranchSpectralModel 类已经在自己的文件中定义并注册了
# 这里不需要重复定义