# --- 自动发现并导入此目录下的所有模块以触发注册 ---
import os
import importlib

# 获取当前文件的目录路径
current_dir = os.path.dirname(__file__)

# 遍历目录下的所有文件
for filename in os.listdir(current_dir):
    # 确保是Python文件，且不是__init__.py自身
    if filename.endswith('.py') and not filename.startswith('__'):
        # 从文件名中获取模块名
        module_name = filename[:-3]
        # 动态地、相对地导入模块，这会执行文件顶层的代码，触发注册
        importlib.import_module(f".{module_name}", package=__name__)

# 从中央注册表导出，方便其他模块调用
from ...registries import FUSION_REGISTRY, register_fusion, HEAD_REGISTRY, register_head