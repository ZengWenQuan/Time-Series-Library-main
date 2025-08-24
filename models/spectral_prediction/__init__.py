import os
import importlib

# 自动发现并导入此目录下的所有模型，以触发注册
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith('.py') and not filename.startswith('__'):
        module_name = filename[:-3]
        importlib.import_module(f".{module_name}", package=__name__)
