
# ==============================================================================
# 中央模型注册系统
# Central Model & Sub-module Registration System
#
# 这个文件定义了项目中所有可插拔模块的注册表（REGISTRY）和
# 对应的注册函数装饰器（register_...）。
# ==============================================================================

# --- 主模型注册表 ---
MODEL_REGISTRY = {}

def register_model(cls):
    """A decorator to register a new main model class using its own name."""
    name = cls.__name__
    if name in MODEL_REGISTRY:
        raise ValueError(f"主模型 '{name}' 已经存在了！不能重复注册")
    MODEL_REGISTRY[name] = cls
    return cls

# --- 连续谱分支注册表 ---
CONTINUUM_BRANCH_REGISTRY = {}

def register_continuum_branch(cls):
    """A decorator to register a new continuum branch class using its own name."""
    name = cls.__name__
    if name in CONTINUUM_BRANCH_REGISTRY:
        raise ValueError(f"Continuum branch '{name}' already registered.")
    CONTINUUM_BRANCH_REGISTRY[name] = cls
    return cls

# --- 归一化谱分支注册表 ---
NORMALIZED_BRANCH_REGISTRY = {}

def register_normalized_branch(cls):
    """A decorator to register a new normalized branch class using its own name."""
    name = cls.__name__
    if name in NORMALIZED_BRANCH_REGISTRY:
        raise ValueError(f"Normalized branch '{name}' already registered.")
    NORMALIZED_BRANCH_REGISTRY[name] = cls
    return cls

# --- 特征融合模块注册表 ---
FUSION_REGISTRY = {}

def register_fusion(cls):
    """A decorator to register a new fusion module class using its own name."""
    name = cls.__name__
    if name in FUSION_REGISTRY:
        raise ValueError(f"Fusion module '{name}' already registered.")
    FUSION_REGISTRY[name] = cls
    return cls

# --- 预测头模块注册表 ---
HEAD_REGISTRY = {}

def register_head(cls):
    """A decorator to register a new head module class using its own name."""
    name = cls.__name__
    if name in HEAD_REGISTRY:
        raise ValueError(f"Head module '{name}' already registered.")
    HEAD_REGISTRY[name] = cls
    return cls
