# 导入子模块包，触发其__init__.py中的动态注册
from . import submodules
from .submodules import global_branches
from .submodules import local_branches
from .submodules import fusion_heads
# 导入新增的模块包
from . import backbones
from . import blocks
from . import heads
# 导入光谱预测模型包，触发其__init__.py中的动态注册
from . import spectral_prediction
