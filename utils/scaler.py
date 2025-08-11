from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
class Scaler:
    """
    标签缩放类，支持StandardScaler、MinMaxScaler和RobustScaler三种方法
    """
    def __init__(self, scaler_type='standard', stats_dict=None, target_names=None):
        """
        初始化标签缩放器
        
        Args:
            scaler_type: 缩放器类型，可选'standard'、'minmax'或'robust'
            stats_dict: (可选) 包含预计算统计数据的字典。键是目标名称，值是包含统计信息的字典。
            target_names: (可选) 需要使用的目标名称列表，用于从stats_dict中筛选和排序。
        """
        self.scaler_type = scaler_type
        self.target_names = target_names
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
            if stats_dict and self.target_names:
                means = [stats_dict[name]['mean'] for name in self.target_names if name in stats_dict]
                std_devs = [stats_dict[name]['std_dev'] for name in self.target_names if name in stats_dict]
                if len(means) == len(self.target_names):
                    print("targets:",target_names,'scaler_type:',scaler_type)
                    print(means)
                    print(std_devs)
                    self.scaler.mean_ = np.array(means)
                    self.scaler.scale_ = np.array(std_devs)
        
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            if stats_dict and self.target_names:
                mins = [stats_dict[name]['min'] for name in self.target_names if name in stats_dict]
                maxs = [stats_dict[name]['max'] for name in self.target_names if name in stats_dict]
                if len(mins) == len(self.target_names):
                    data_min = np.array(mins)
                    data_max = np.array(maxs)
                    self.scaler.scale_ = 1.0 / (data_max - data_min)
                    self.scaler.min_ = -data_min * self.scaler.scale_
                    self.scaler.data_min_ = data_min
                    self.scaler.data_max_ = data_max
                    self.scaler.data_range_ = data_max - data_min

        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
            if stats_dict and self.target_names:
                centers = [stats_dict[name]['median_50th_percentile'] for name in self.target_names if name in stats_dict]
                q1s = [stats_dict[name]['25th_percentile'] for name in self.target_names if name in stats_dict]
                q3s = [stats_dict[name]['75th_percentile'] for name in self.target_names if name in stats_dict]
                if len(centers) == len(self.target_names):
                    self.scaler.center_ = np.array(centers)
                    self.scaler.scale_ = np.array(q3s) - np.array(q1s)
        elif scaler_type is not None:
            raise ValueError(f"不支持的缩放器类型: {scaler_type}")

    def fit(self, data, target_names=None):
        """
        拟合标签缩放器
        
        Args:
            data: 形状为[samples, features]的数据
            target_names: (可选) 目标名称列表
        """
        if target_names:
            self.target_names = target_names
        return self.scaler.fit(data)
    
    def transform(self, data):
        """
        转换数据
        
        Args:
            data: 形状为[samples, features]的数据
        
        Returns:
            转换后的数据
        """
        return self.scaler.transform(data)
    
    def fit_transform(self, data):
        """
        拟合并转换数据
        
        Args:
            data: 形状为[samples, features]的数据
        
        Returns:
            转换后的数据
        """
        return self.scaler.fit_transform(data)
    
    def inverse_transform(self, data):
        """
        将缩放后的数据转换回原始尺度
        
        Args:
            data: 形状为[samples, features]的缩放后数据
        
        Returns:
            原始尺度的数据
        """
        return self.scaler.inverse_transform(data)
