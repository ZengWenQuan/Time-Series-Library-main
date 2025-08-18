from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

class Scaler:
    """
    一个通用的缩放类，可以处理两种场景：
    1. 标签（多列）分别归一化。
    2. 光谱特征（多维数组）使用单一统计值（如flux）进行全局归一化。
    """
    def __init__(self, scaler_type='standard', stats_dict=None, target_names=None):
        """
        初始化缩放器
        
        Args:
            scaler_type: 缩放器类型，可选'standard'、'minmax'或'robust'
            stats_dict: (可选) 包含预计算统计数据的字典。
            target_names: (可选) 需要使用的目标名称列表。
        """
        self.scaler_type = scaler_type
        self.target_names = target_names
        self.scaler = None  # 用于存储scikit-learn的scaler实例（用于标签）
        self.feature_stats = None  # 用于存储单一特征（如flux）的统计数据

        # 如果没有提供预计算的统计数据，则初始化一个空的scaler，等待后续fit
        if not stats_dict or not target_names:
            if scaler_type == 'standard': self.scaler = StandardScaler()
            elif scaler_type == 'minmax': self.scaler = MinMaxScaler()
            elif scaler_type == 'robust': self.scaler = RobustScaler()
            return

        # --- 检查所有请求的target是否存在于统计字典中，防止静默失败 ---
        cleaned_target_names = [str(n).strip() for n in self.target_names]
        print(self.target_names)
        cleaned_stats_keys = {str(k).strip() for k in stats_dict.keys()}
        found_targets = [name  for name in cleaned_target_names if name in cleaned_stats_keys]
        if len(found_targets) != len(self.target_names):
            print(stats_dict.keys())
            raise ValueError(f"Scaler初始化失败：在 stats.yaml 中不匹配\n target:{cleaned_target_names}\n stats.yaml:{cleaned_stats_keys}\n{found_targets}")

        # --- 场景1: 特征归一化 (例如, target_names=['flux']) ---
        # 对整个多维数组使用单一的统计值
        if len(self.target_names) == 1:
            feature_name = self.target_names[0]
            self.feature_stats = stats_dict[feature_name]
            print(f"初始化为特征缩放模式，使用 '{feature_name}' 的统计数据。")

        # --- 场景2: 标签归一化 (例如, target_names=['Teff', 'logg']) ---
        # 对每一列使用其各自的统计值
        else:
            print(f"初始化为标签缩放模式，处理: {self.target_names}")
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array([stats_dict[name]['mean'] for name in self.target_names])
                self.scaler.scale_ = np.array([stats_dict[name]['std_dev'] for name in self.target_names])
            
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
                mins = np.array([stats_dict[name]['min'] for name in self.target_names])
                maxs = np.array([stats_dict[name]['max'] for name in self.target_names])
                self.scaler.scale_ = 1.0 / (maxs - mins)
                self.scaler.min_ = -mins * self.scaler.scale_
                self.scaler.data_min_ = mins
                self.scaler.data_max_ = maxs
                self.scaler.data_range_ = maxs - mins

            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
                self.scaler.center_ = np.array([stats_dict[name]['median_50th_percentile'] for name in self.target_names])
                q1s = np.array([stats_dict[name]['25th_percentile'] for name in self.target_names])
                q3s = np.array([stats_dict[name]['75th_percentile'] for name in self.target_names])
                self.scaler.scale_ = q3s - q1s
            
            elif scaler_type is not None:
                raise ValueError(f"不支持的缩放器类型: {scaler_type}")

    def fit(self, data, target_names=None):
        if self.feature_stats is not None:
            raise NotImplementedError("无法对已从stats文件初始化的'特征缩放器'进行fit操作。")
        if self.scaler is None:
            raise ValueError("Scaler未初始化，无法fit。")
        if target_names:
            self.target_names = target_names
        return self.scaler.fit(data)

    def transform(self, data):
        if self.feature_stats is not None:
            # 特征归一化：直接进行数学运算
            if self.scaler_type == 'standard':
                return (data - self.feature_stats['mean']) / self.feature_stats['std_dev']
            else:
                raise NotImplementedError(f"特征归一化暂不支持'{self.scaler_type}'类型。")
        elif self.scaler is not None and hasattr(self.scaler, 'mean_'):
            # 标签归一化：使用scikit-learn的scaler
            return self.scaler.transform(data)
        else:
            raise RuntimeError("Scaler未被正确初始化或fit。请先调用fit或使用stats_dict初始化。")

    def fit_transform(self, data):
        if self.feature_stats is not None:
            raise NotImplementedError("无法对已从stats文件初始化的'特征缩放器'进行fit_transform操作。")
        if self.scaler is None:
            raise ValueError("Scaler未初始化，无法fit_transform。")
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        if self.feature_stats is not None:
            # 特征反归一化
            if self.scaler_type == 'standard':
                return (data * self.feature_stats['std_dev']) + self.feature_stats['mean']
            else:
                raise NotImplementedError(f"特征反归一化暂不支持'{self.scaler_type}'类型。")
        elif self.scaler is not None and hasattr(self.scaler, 'mean_'):
            # 标签反归一化
            return self.scaler.inverse_transform(data)
        else:
            raise RuntimeError("Scaler未被正确初始化或fit。")