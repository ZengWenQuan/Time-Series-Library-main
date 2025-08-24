# 导入子模块包，触发其__init__.py中的动态注册
from . import submodules
from .submodules import continuum_branches
from .submodules import normalized_branches
from .submodules import fusion_heads
# 导入光谱预测模型包，触发其__init__.py中的动态注册
from . import spectral_prediction
