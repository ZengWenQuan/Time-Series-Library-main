from ..registries import register_model
from .GenericSpectralModel import GenericSpectralModel

@register_model
class FlexibleFusionNet(GenericSpectralModel):
    """
    A flexible fusion network that inherits all functionality directly from
    GenericSpectralModel. It is registered under its own name to allow for
    separate configuration and tracking, while reusing the generic model logic.
    """
    pass