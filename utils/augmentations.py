import numpy as np
import random

# --- 数据增强注册器 ---
AUGMENTATION_REGISTRY = {}
def register_augmentation(name):
    def decorator(cls):
        if name in AUGMENTATION_REGISTRY: raise ValueError(f"Augmentation '{name}' already registered.")
        AUGMENTATION_REGISTRY[name] = cls
        return cls
    return decorator

# --- 带概率的增强包装器 ---
class ProbabilisticAugmentation:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p
    def __call__(self, x):
        if random.random() < self.p:
            return self.transform(x)
        return x

# --- 增强流程控制器 (原Compose类) ---
class Transforms:
    """根据配置构建并执行一个数据增强流水线。"""
    def __init__(self, augs_conf):
        self.augmentations = []
        if not augs_conf: # 如果配置为空，则不进行任何操作
            return

        for aug_conf in augs_conf:
            # 检查开关是否为true
            if aug_conf.get('enabled', False):
                # 从注册器中查找对应的增强类
                if aug_conf['name'] in AUGMENTATION_REGISTRY:
                    AugmentationClass = AUGMENTATION_REGISTRY[aug_conf['name']]
                    # 使用配置中的参数初始化增强类
                    transform = AugmentationClass(**aug_conf.get('params', {}))
                    # 使用概率包装器
                    prob_transform = ProbabilisticAugmentation(
                        transform,
                        p=aug_conf.get('p', 0.5)
                    )
                    self.augmentations.append(prob_transform)

    def __call__(self, x):
        for aug in self.augmentations:
            x = aug(x)
        return x

# --- 基类 ---
class Augmentation:
    def __call__(self, x):
        raise NotImplementedError

# --- 具体实现 ---
@register_augmentation('add_noise')
class AddNoise(Augmentation):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, x):
        return x + np.random.normal(0, self.sigma, size=x.shape)

@register_augmentation('random_shift')
class RandomShift(Augmentation):
    def __init__(self, max_shift=2):
        self.max_shift = max_shift
    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift)

@register_augmentation('random_scaling')
class RandomScaling(Augmentation):
    def __init__(self, min_scale=0.9, max_scale=1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale
    def __call__(self, x):
        return x * np.random.uniform(self.min_scale, self.max_scale)

@register_augmentation('random_masking')
class RandomMasking(Augmentation):
    def __init__(self, mask_width_ratio=0.1, mask_value=1.0):
        self.mask_width_ratio = mask_width_ratio
        self.mask_value = mask_value
    def __call__(self, x):
        new_x = x.copy()
        mask_width = int(len(x) * self.mask_width_ratio)
        if mask_width > 0:
            start_index = np.random.randint(0, len(x) - mask_width)
            new_x[start_index : start_index + mask_width] = self.mask_value
        return new_x