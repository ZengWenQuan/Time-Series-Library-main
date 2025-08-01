import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import random

def classify_feh(feh_value):
    """
    根据FeH值进行分类
    EMP (Extremely Metal-Poor): [Fe/H] < -3.0
    VMP (Very Metal-Poor): -3.0 <= [Fe/H] < -2.0
    MP (Metal-Poor): -2.0 <= [Fe/H] < -1.0
    Normal: [Fe/H] >= -1.0
    """
    if feh_value < -3.0:
        return "EMP"
    elif feh_value < -2.0:
        return "VMP"
    elif feh_value < -1.0:
        return "MP"
    else:
        return "Normal"

class StellarSampling:
    """
    基于FeH分类的过采样和欠采样方法
    适用于恒星参数估计的回归任务
    """
    def __init__(self, sampling_strategy='auto', random_state=42, k_neighbors=5, feh_index=2):
        """
        初始化
        
        参数:
        sampling_strategy: 采样策略，可以是'auto'、'balanced'或者一个字典
                          'auto': 所有类别采样到与最多类别相同数量
                          'balanced': 所有类别采样到相同数量
                          dict: 指定每个类别的目标数量，如{'EMP': 100, 'VMP': 200, 'MP': 300, 'Normal': 400}
        random_state: 随机种子
        k_neighbors: KNN中的邻居数量，用于SMOTE过采样
        feh_index: FeH在标签数组中的索引位置
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.feh_index = feh_index
        np.random.seed(random_state)
        random.seed(random_state)
    
    def fit_resample(self, X, y):
        """
        对数据进行过采样和欠采样
        
        参数:
        X: 特征数据，形状为(n_samples, n_features)
        y: 标签数据，形状为(n_samples, n_labels)
        
        返回:
        X_resampled: 重采样后的特征数据
        y_resampled: 重采样后的标签数据
        """
        # 获取FeH值并分类
        feh_values = y[:, self.feh_index]
        categories = np.array([classify_feh(feh) for feh in feh_values])
        
        # 统计各类别数量
        class_counts = Counter(categories)
        print(f"原始数据类别分布: {class_counts}")
        
        # 确定目标采样数量
        if self.sampling_strategy == 'auto':
            max_count = max(class_counts.values())
            target_counts = {cls: max_count for cls in class_counts.keys()}
        elif self.sampling_strategy == 'balanced':
            total_samples = sum(class_counts.values())
            n_classes = len(class_counts)
            target_count = total_samples // n_classes
            target_counts = {cls: target_count for cls in class_counts.keys()}
        elif isinstance(self.sampling_strategy, dict):
            target_counts = self.sampling_strategy
        else:
            raise ValueError("sampling_strategy必须是'auto'、'balanced'或者一个字典")
        
        # 进行重采样
        X_resampled = []
        y_resampled = []
        
        for cls in class_counts.keys():
            # 获取当前类别的样本
            cls_indices = np.where(categories == cls)[0]
            X_cls = X[cls_indices]
            y_cls = y[cls_indices]
            
            current_count = len(cls_indices)
            target_count = target_counts.get(cls, current_count)
            
            if target_count <= current_count:  # 欠采样
                # 随机选择样本
                selected_indices = np.random.choice(current_count, target_count, replace=False)
                X_cls_resampled = X_cls[selected_indices]
                y_cls_resampled = y_cls[selected_indices]
            else:  # 过采样
                # 使用SMOTE思想进行过采样
                X_cls_resampled, y_cls_resampled = self._oversample(X_cls, y_cls, target_count)
            
            X_resampled.append(X_cls_resampled)
            y_resampled.append(y_cls_resampled)
        
        # 合并重采样后的数据
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.vstack(y_resampled)
        
        # 随机打乱数据
        indices = np.arange(len(X_resampled))
        np.random.shuffle(indices)
        X_resampled = X_resampled[indices]
        y_resampled = y_resampled[indices]
        
        # 输出重采样后的类别分布
        new_feh_values = y_resampled[:, self.feh_index]
        new_categories = np.array([classify_feh(feh) for feh in new_feh_values])
        new_class_counts = Counter(new_categories)
        print(f"重采样后数据类别分布: {new_class_counts}")
        
        return X_resampled, y_resampled
    
    def _oversample(self, X, y, target_count):
        """
        使用SMOTE思想进行过采样
        
        参数:
        X: 特征数据
        y: 标签数据
        target_count: 目标样本数量
        
        返回:
        X_oversampled: 过采样后的特征数据
        y_oversampled: 过采样后的标签数据
        """
        n_samples = X.shape[0]
        
        # 如果样本数量小于k_neighbors，则调整k_neighbors
        k = min(self.k_neighbors, n_samples - 1)
        if k <= 1:  # 如果样本太少，使用简单复制
            n_samples_to_generate = target_count - n_samples
            indices = np.random.randint(0, n_samples, size=n_samples_to_generate)
            X_new = X[indices]
            y_new = y[indices]
        else:
            # 使用KNN找到每个样本的邻居
            nn = NearestNeighbors(n_neighbors=k+1).fit(X)
            distances, indices = nn.kneighbors(X)
            
            # 生成新样本
            X_new = []
            y_new = []
            n_samples_to_generate = target_count - n_samples
            
            for _ in range(n_samples_to_generate):
                # 随机选择一个样本
                i = np.random.randint(0, n_samples)
                # 随机选择一个邻居（跳过自身）
                nn_index = indices[i, 1 + np.random.randint(0, k)]
                
                # 在样本和邻居之间插值
                alpha = np.random.random()
                X_interpolated = X[i] + alpha * (X[nn_index] - X[i])
                y_interpolated = y[i] + alpha * (y[nn_index] - y[i])
                
                X_new.append(X_interpolated)
                y_new.append(y_interpolated)
            
            X_new = np.array(X_new)
            y_new = np.array(y_new)
        
        # 合并原始样本和新生成的样本
        X_oversampled = np.vstack([X, X_new])
        y_oversampled = np.vstack([y, y_new])
        
        return X_oversampled, y_oversampled

def apply_stellar_sampling(X, y, sampling_strategy='balanced', random_state=42, k_neighbors=5, feh_index=2):
    """
    应用恒星采样方法的便捷函数
    
    参数:
    X: 特征数据，形状为(n_samples, n_features)
    y: 标签数据，形状为(n_samples, n_labels)
    sampling_strategy: 采样策略
    random_state: 随机种子
    k_neighbors: KNN中的邻居数量
    feh_index: FeH在标签数组中的索引位置
    
    返回:
    X_resampled: 重采样后的特征数据
    y_resampled: 重采样后的标签数据
    """
    sampler = StellarSampling(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
        feh_index=feh_index
    )
    return sampler.fit_resample(X, y) 