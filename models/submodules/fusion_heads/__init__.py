# A registry to store fusion classes
FUSION_REGISTRY = {}

def register_fusion(name):
    """A decorator to register a new fusion class."""
    def decorator(cls):
        if name in FUSION_REGISTRY:
            raise ValueError(f"Fusion module '{name}' already registered.")
        FUSION_REGISTRY[name] = cls
        return cls
    return decorator

# A registry to store head classes
HEAD_REGISTRY = {}

def register_head(name):
    """A decorator to register a new head class."""
    def decorator(cls):
        if name in HEAD_REGISTRY:
            raise ValueError(f"Head module '{name}' already registered.")
        HEAD_REGISTRY[name] = cls
        return cls
    return decorator

# --- Import all modules in this directory to trigger registration ---
from . import fusion_modules
from . import prediction_heads
