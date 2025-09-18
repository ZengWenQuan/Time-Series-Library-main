
# ==============================================================================
# 简化的模块注册系统
# Simplified Module Registration System
#
# 统一的注册机制，所有模块都使用类名直接注册
# ==============================================================================

from typing import Type, Dict
import torch.nn as nn

# 统一注册器字典
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
BACKBONES: Dict[str, Type[nn.Module]] = {}
GLOBAL_BRANCH_REGISTRY: Dict[str, Type[nn.Module]] = {}
LOCAL_BRANCH_REGISTRY: Dict[str, Type[nn.Module]] = {}
FUSION_REGISTRY: Dict[str, Type[nn.Module]] = {}
HEAD_REGISTRY: Dict[str, Type[nn.Module]] = {}
BLOCKS: Dict[str, Type[nn.Module]] = {}

# 兼容性别名
HEADS = HEAD_REGISTRY
MODELS = MODEL_REGISTRY
# 保留旧名称以确保向后兼容
CONTINUUM_BRANCH_REGISTRY = GLOBAL_BRANCH_REGISTRY
NORMALIZED_BRANCH_REGISTRY = LOCAL_BRANCH_REGISTRY

def _register_to_registry(registry: Dict[str, Type[nn.Module]], cls: Type[nn.Module]):
    """通用注册函数，使用类名作为键"""
    name = cls.__name__
    if name in registry:
        raise ValueError(f"模块 '{name}' 已经存在于注册表中！不能重复注册")
    registry[name] = cls
    return cls

# === 主模型注册装饰器 ===
def register_model(name_or_cls=None):
    """
    主模型注册装饰器
    用法:
        @register_model
        class MyModel(nn.Module): ...
    或
        @register_model('CustomName')
        class MyModel(nn.Module): ...
    """
    def decorator(cls):
        # 如果提供了自定义名称，使用它；否则使用类名
        registry_name = name_or_cls if isinstance(name_or_cls, str) else cls.__name__
        if registry_name in MODEL_REGISTRY:
            raise ValueError(f"主模型 '{registry_name}' 已经存在了！不能重复注册")
        MODEL_REGISTRY[registry_name] = cls
        return cls

    # 处理两种用法：@register_model 和 @register_model('name')
    if isinstance(name_or_cls, type):
        # 直接使用类名注册
        return _register_to_registry(MODEL_REGISTRY, name_or_cls)
    else:
        # 返回装饰器函数
        return decorator

# === 简化的注册装饰器 ===
def register_backbone(cls: Type[nn.Module]):
    """主干网络注册装饰器"""
    return _register_to_registry(BACKBONES, cls)

def register_global_branch(cls: Type[nn.Module]):
    """全局分支注册装饰器"""
    _register_to_registry(GLOBAL_BRANCH_REGISTRY, cls)
    # 同时注册到BLOCKS以保持兼容性
    return _register_to_registry(BLOCKS, cls)

def register_local_branch(cls: Type[nn.Module]):
    """局部分支注册装饰器"""
    _register_to_registry(LOCAL_BRANCH_REGISTRY, cls)
    # 同时注册到BLOCKS以保持兼容性
    return _register_to_registry(BLOCKS, cls)

# === 向后兼容的别名 ===
def register_continuum_branch(cls: Type[nn.Module]):
    """连续谱分支注册装饰器（向后兼容，推荐使用register_global_branch）"""
    return register_global_branch(cls)

def register_normalized_branch(cls: Type[nn.Module]):
    """归一化谱分支注册装饰器（向后兼容，推荐使用register_local_branch）"""
    return register_local_branch(cls)

def register_fusion(cls: Type[nn.Module]):
    """特征融合模块注册装饰器"""
    return _register_to_registry(FUSION_REGISTRY, cls)

def register_head(cls: Type[nn.Module]):
    """预测头模块注册装饰器"""
    return _register_to_registry(HEAD_REGISTRY, cls)

def register_block(cls: Type[nn.Module]):
    """通用块注册装饰器（用于自定义组件）"""
    return _register_to_registry(BLOCKS, cls)

# === 辅助函数 ===
def list_registered_models():
    """列出所有已注册的模型"""
    return {
        'models': list(MODEL_REGISTRY.keys()),
        'backbones': list(BACKBONES.keys()),
        'global_branches': list(GLOBAL_BRANCH_REGISTRY.keys()),
        'local_branches': list(LOCAL_BRANCH_REGISTRY.keys()),
        'fusion_modules': list(FUSION_REGISTRY.keys()),
        'heads': list(HEAD_REGISTRY.keys()),
        'blocks': list(BLOCKS.keys())
    }

def get_model_class(model_name: str):
    """根据名称获取模型类"""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    raise ValueError(f"模型 '{model_name}' 未注册。可用模型: {list(MODEL_REGISTRY.keys())}")