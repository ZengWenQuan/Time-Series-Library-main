
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..registries import     register_model
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