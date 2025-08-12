# A registry to store normalized branch classes
NORMALIZED_BRANCH_REGISTRY = {}

def register_normalized_branch(cls):
    """A decorator to register a new normalized branch class using its own name."""
    name = cls.__name__
    if name in NORMALIZED_BRANCH_REGISTRY:
        raise ValueError(f"Normalized branch '{name}' already registered.")
    NORMALIZED_BRANCH_REGISTRY[name] = cls
    return cls

# --- 新增：导入此目录下的所有分支模块，以触发注册 ---
from . import attention_branch
from . import msp_branch
from . import upsample_branch
from . import pyramid_extractor
from . import multiscale_cnn_branch