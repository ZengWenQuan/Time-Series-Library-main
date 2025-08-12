# A registry to store continuum branch classes
CONTINUUM_BRANCH_REGISTRY = {}

def register_continuum_branch(cls):
    """A decorator to register a new continuum branch class using its own name."""
    name = cls.__name__
    if name in CONTINUUM_BRANCH_REGISTRY:
        raise ValueError(f"Continuum branch '{name}' already registered.")
    CONTINUUM_BRANCH_REGISTRY[name] = cls
    return cls

# --- 新增：导入此目录下的所有分支模块，以触发注册 ---
from . import large_kernel_branch
from . import moe_branch
from . import wavelet_branch
from . import cnn_transformer_branch
