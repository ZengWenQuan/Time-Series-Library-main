import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差(MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

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

def calculate_metrics(pred, true, param_names=None):
    """
    计算多个指标，并为每个参数单独计算
    
    Args:
        pred: 预测值，形状为 [samples, n_params]
        true: 真实值，形状为 [samples, n_params]
        param_names: 参数名称列表，默认为 ['Teff', 'logg', 'FeH', 'CFe']
    
    Returns:
        metrics_dict: 包含各种指标的字典
    """
    if param_names is None:
        param_names = ['Teff', 'logg', 'FeH', 'CFe']
    
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    # 初始化指标字典
    metrics_dict = {}
    
    # 计算整体指标
    mae_all = mean_absolute_error(true, pred)
    mse_all = mean_squared_error(true, pred)
    rmse_all = np.sqrt(mse_all)
    
    metrics_dict['mae'] = mae_all
    metrics_dict['mse'] = mse_all
    metrics_dict['rmse'] = rmse_all
    
    # 为每个参数计算单独的指标
    n_params = pred.shape[1]
    for i in range(n_params):
        param_name = param_names[i] if i < len(param_names) else f'param_{i}'
        
        # 提取当前参数的预测值和真实值
        param_pred = pred[:, i]
        param_true = true[:, i]
        
        # 计算各种指标
        mae = mean_absolute_error(param_true, param_pred)
        mse = mean_squared_error(param_true, param_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(param_true, param_pred)
        
        # 计算MAPE，处理可能的除零错误
        try:
            mape = mean_absolute_percentage_error(param_true, param_pred)
        except:
            mape = np.nan
        
        # 将指标添加到字典中
        metrics_dict[f'{param_name}_mae'] = mae
        metrics_dict[f'{param_name}_mse'] = mse
        metrics_dict[f'{param_name}_rmse'] = rmse
        metrics_dict[f'{param_name}_r2'] = r2
        metrics_dict[f'{param_name}_mape'] = mape
        
        # 保存预测值和真实值以便后续绘图
        metrics_dict[f'{param_name}_pred'] = param_pred
        metrics_dict[f'{param_name}_true'] = param_true
    
    return metrics_dict

def format_metrics(metrics_dict):
    """
    将指标字典格式化为表格形式的可读字符串，所有列宽度一致
    """
    # 首先格式化整体指标
    result = f"Overall: mae:{metrics_dict['mae']:.4f}, mse:{metrics_dict['mse']:.4f}, rmse:{metrics_dict['rmse']:.4f}\n\n"
    
    # 提取参数名称（通过查找字典键中的模式）
    param_names = set()
    # 定义指标顺序
    metrics_types = ['mae', 'mse', 'rmse', 'r2', 'mape']
    
    for key in metrics_dict.keys():
        if '_' in key:
            param_name = key.split('_')[0]
            if param_name not in ['param'] and key.split('_')[1] in metrics_types:
                param_names.add(param_name)
    
    # 设置统一的列宽
    col_width = 9  # "Parameter"的长度
    
    # 创建表头行
    header = f"{'Parameter':{col_width}} |"
    for metric in metrics_types:
        if metric == 'mape':
            header += f" {'MAPE(%)':{col_width}} |"
        else:
            header += f" {metric.upper():{col_width}} |"
    
    # 添加表头
    result += header + "\n"
    # 添加足够长的分隔线
    result += "-" * len(header) + "\n"
    
    # 为每个参数添加一行
    for param in sorted(param_names):
        row = f"{param:{col_width}} |"
        
        for metric in metrics_types:
            key = f"{param}_{metric}"
            if key in metrics_dict:
                # 为每种指标调整格式
                if metric == 'mape':
                    formatted = f"{metrics_dict[key]:.2f}%"
                    row += f" {formatted:{col_width}} |"
                elif metric == 'r2':
                    formatted = f"{metrics_dict[key]:.4f}"
                    row += f" {formatted:{col_width}} |"
                elif metric == 'mae' or metric == 'rmse':
                    # 对于可能有大值的指标，使用更灵活的格式
                    value = metrics_dict[key]
                    if abs(value) < 0.01:
                        formatted = f"{value:.4f}"
                    elif abs(value) < 10:
                        formatted = f"{value:.4f}"
                    else:
                        formatted = f"{value:.4f}"
                    row += f" {formatted:{col_width}} |"
                else:
                    # MSE可能有非常大的值
                    value = metrics_dict[key]
                    if abs(value) < 0.01:
                        formatted = f"{value:.4f}"
                    elif abs(value) < 10:
                        formatted = f"{value:.4f}"
                    elif abs(value) < 1000:
                        formatted = f"{value:.2f}"
                    else:
                        formatted = f"{value:.2f}"
                    row += f" {formatted:{col_width}} |"
            else:
                row += f" {'N/A':{col_width}} |"
        
        result += row + "\n"
    
    return result

def classify_feh(feh_value):
    """
    根据FeH值对恒星进行分类
    
    Args:
        feh_value: FeH值
    
    Returns:
        分类结果: 0=EMP, 1=VMP, 2=MP, 3=正常恒星
    """
    if feh_value <= -3:
        return 0  # EMP (Extremely Metal-Poor)
    elif -3 < feh_value <= -2:
        return 1  # VMP (Very Metal-Poor)
    elif -2 < feh_value <= -1:
        return 2  # MP (Metal-Poor)
    else:
        return 3  # 正常恒星
        
def calculate_feh_classification_metrics(pred, true, feh_index=2):
    """
    计算FeH分类指标
    
    Args:
        pred: 预测值，形状为 [samples, n_params]
        true: 真实值，形状为 [samples, n_params]
        feh_index: FeH在参数中的索引，默认为2
    
    Returns:
        metrics_dict: 包含分类指标的字典
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    # 提取FeH值
    feh_pred = pred[:, feh_index]
    feh_true = true[:, feh_index]
    
    # 将连续值转换为分类
    y_true = np.array([classify_feh(val) for val in feh_true])
    y_pred = np.array([classify_feh(val) for val in feh_pred])
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # 计算各类别的指标
    class_names = ['EMP', 'VMP', 'MP', 'Normal']
    
    # 初始化指标字典
    metrics_dict = {}
    
    # 计算整体准确率
    metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 计算宏平均指标
    metrics_dict['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics_dict['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算加权平均指标
    metrics_dict['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics_dict['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics_dict['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 计算每个类别的指标
    for i, class_name in enumerate(class_names):
        # 使用二分类方式计算每个类别的指标
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        # 处理可能的零除错误
        if np.sum(y_true_binary) > 0:
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        metrics_dict[f'{class_name}_precision'] = precision
        metrics_dict[f'{class_name}_recall'] = recall
        metrics_dict[f'{class_name}_f1'] = f1
    
    # 添加混淆矩阵
    metrics_dict['confusion_matrix'] = cm
    metrics_dict['class_names'] = class_names
    
    # 保存原始预测值和真实值
    metrics_dict['feh_pred'] = feh_pred
    metrics_dict['feh_true'] = feh_true
    metrics_dict['y_pred'] = y_pred
    metrics_dict['y_true'] = y_true
    
    return metrics_dict

def format_feh_classification_metrics(metrics_dict):
    """
    将FeH分类指标字典格式化为表格形式的可读字符串
    
    Args:
        metrics_dict: 包含分类指标的字典
    
    Returns:
        formatted_str: 格式化后的字符串
    """
    # 设置统一的列宽
    col_width = 9
    
    # 格式化整体指标
    result = "FeH Classification Metrics:\n\n"
    
    # 添加整体准确率
    result += f"Overall Accuracy: {metrics_dict['accuracy']:.4f}\n\n"
    
    # 添加宏平均和加权平均指标
    result += "Macro and Weighted Averages:\n"
    header = f"{'Average':{col_width}} |"
    for metric in ['precision', 'recall', 'f1']:
        header += f" {metric.title():{col_width}} |"
    
    result += header + "\n"
    result += "-" * len(header) + "\n"
    
    # 宏平均行
    row = f"{'Macro':{col_width}} |"
    for metric in ['precision', 'recall', 'f1']:
        key = f'macro_{metric}'
        formatted = f"{metrics_dict[key]:.4f}"
        row += f" {formatted:{col_width}} |"
    result += row + "\n"
    
    # 加权平均行
    row = f"{'Weighted':{col_width}} |"
    for metric in ['precision', 'recall', 'f1']:
        key = f'weighted_{metric}'
        formatted = f"{metrics_dict[key]:.4f}"
        row += f" {formatted:{col_width}} |"
    result += row + "\n\n"
    
    # 添加每个类别的指标
    result += "Per-Class Metrics:\n"
    header = f"{'Class':{col_width}} |"
    for metric in ['precision', 'recall', 'f1']:
        header += f" {metric.title():{col_width}} |"
    
    result += header + "\n"
    result += "-" * len(header) + "\n"
    
    # 每个类别的行
    class_names = metrics_dict['class_names']
    for class_name in class_names:
        row = f"{class_name:{col_width}} |"
        for metric in ['precision', 'recall', 'f1']:
            key = f'{class_name}_{metric}'
            formatted = f"{metrics_dict[key]:.4f}"
            row += f" {formatted:{col_width}} |"
        result += row + "\n"
    
    # 添加混淆矩阵
    result += "\nConfusion Matrix:\n"
    cm = metrics_dict['confusion_matrix']
    
    # 创建混淆矩阵表头
    cm_header = f"{'True/Pred':{col_width}} |"
    for class_name in class_names:
        cm_header += f" {class_name:{col_width}} |"
    
    result += cm_header + "\n"
    result += "-" * len(cm_header) + "\n"
    
    # 添加混淆矩阵行
    for i, class_name in enumerate(class_names):
        row = f"{class_name:{col_width}} |"
        for j in range(len(class_names)):
            formatted = f"{cm[i, j]}"
            row += f" {formatted:{col_width}} |"
        result += row + "\n"
    
    return result

def save_regression_metrics(metrics_dict, save_dir, param_names=None, phase="test"):
    """
    保存回归指标到CSV文件，并生成预测值与真实值的对比图（矢量图PDF格式），图中包含关键指标。
    
    Args:
        metrics_dict: 包含各种指标的字典
        save_dir: 保存目录
        param_names: 参数名称列表
        phase: 阶段名称，如'best'或'last'
    """
    if not save_dir:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if param_names is None:
        param_names = ['Teff', 'logg', 'FeH', 'CFe']
    
    # 创建回归指标保存目录
    regression_dir = os.path.join(save_dir, phase, 'regression')
    os.makedirs(regression_dir, exist_ok=True)
    
    # ... (The CSV saving part remains the same) ...
    metrics_file = os.path.join(regression_dir, 'metrics.csv')
    metrics_data = []
    metrics_types = ['mae', 'mse', 'rmse', 'r2', 'mape']
    overall_metrics = {'Parameter': 'Overall'}
    for metric in metrics_types:
        if metric in metrics_dict:
            if metric == 'mape':
                overall_metrics[metric.upper() + '(%)'] = f"{metrics_dict[metric]:.2f}%"
            else:
                overall_metrics[metric.upper()] = f"{metrics_dict[metric]:.4f}"
    metrics_data.append(overall_metrics)
    for param in param_names:
        param_metrics = {'Parameter': param}
        for metric in metrics_types:
            key = f'{param}_{metric}'
            if key in metrics_dict:
                if metric == 'mape':
                    param_metrics[metric.upper() + '(%)'] = f"{metrics_dict[key]:.2f}%"
                else:
                    param_metrics[metric.upper()] = f"{metrics_dict[key]:.4f}"
        metrics_data.append(param_metrics)
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['Parameter'] + [m.upper() + '(%)' if m == 'mape' else m.upper() for m in metrics_types]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_data:
            writer.writerow(row)
    
    # --- Modified Plotting Section ---
    # 创建预测值与真实值的对比图
    num_params = len(param_names)
    # Adjust layout based on number of parameters
    if num_params <= 2:
        n_cols = num_params
        n_rows = 1
        figsize = (8 * n_cols, 6)
    else:
        n_cols = 2
        n_rows = (num_params + 1) // 2
        figsize = (15, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]
        pred_key = f'{param}_pred'
        true_key = f'{param}_true'
        
        if pred_key in metrics_dict and true_key in metrics_dict:
            pred = metrics_dict[pred_key]
            true = metrics_dict[true_key]
            
            ax.scatter(pred, true, alpha=0.5, s=10) # Smaller points for clarity
            
            # 添加对角线
            min_val = min(np.min(pred), np.min(true))
            max_val = max(np.max(pred), np.max(true))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
            
            ax.set_title(f'{param} Prediction vs Truth')
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('True Value')
            ax.grid(True)

            # --- Add Metrics Annotation ---
            mae = metrics_dict.get(f'{param}_mae', float('nan'))
            rmse = metrics_dict.get(f'{param}_rmse', float('nan'))
            r2 = metrics_dict.get(f'{param}_r2', float('nan'))
            
            metrics_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
            
            ax.text(0.95, 0.95, metrics_text,
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

    # Hide any unused subplots
    for j in range(num_params, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # Save as PDF (vector graphics) file
    plt.savefig(os.path.join(regression_dir, 'prediction_vs_truth.pdf'))
    plt.close(fig)

def save_feh_classification_metrics(metrics_dict, save_dir, phase="test"):
    """
    保存FeH分类指标到CSV文件，并生成PDF格式的可视化图表。
    
    Args:
        metrics_dict: 包含分类指标的字典
        save_dir: 保存目录
        phase: 阶段名称，如'best'或'last'
    """
    if not save_dir:
        return
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建分类指标保存目录
    classification_dir = os.path.join(save_dir, phase, 'classification')
    os.makedirs(classification_dir, exist_ok=True)
    
    # ... (CSV saving part remains the same) ...
    metrics_file = os.path.join(classification_dir, 'metrics.csv')
    metrics_data = []
    metrics_data.append({'Metric': 'Overall Accuracy', 'Value': f"{metrics_dict['accuracy']:.4f}"})
    for avg_type in ['macro', 'weighted']:
        for metric in ['precision', 'recall', 'f1']:
            key = f'{avg_type}_{metric}'
            if key in metrics_dict:
                metrics_data.append({'Metric': f'{avg_type.title()} {metric.title()}', 'Value': f"{metrics_dict[key]:.4f}"})
    class_names = metrics_dict['class_names']
    for class_name in class_names:
        for metric in ['precision', 'recall', 'f1']:
            key = f'{class_name}_{metric}'
            if key in metrics_dict:
                metrics_data.append({'Metric': f'{class_name} {metric.title()}', 'Value': f"{metrics_dict[key]:.4f}"})
    with open(metrics_file, 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_data:
            writer.writerow(row)
    cm_file = os.path.join(classification_dir, 'confusion_matrix.csv')
    cm = metrics_dict['confusion_matrix']
    with open(cm_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['True/Pred'] + class_names)
        for i, class_name in enumerate(class_names):
            writer.writerow([class_name] + list(cm[i]))

    # --- Modified Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(classification_dir, 'confusion_matrix.pdf'))
    plt.close(fig)
    
    # --- Modified FeH Prediction vs Truth Plot ---
    if 'feh_pred' in metrics_dict and 'feh_true' in metrics_dict:
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, class_name in enumerate(class_names):
            mask = metrics_dict['y_true'] == i
            ax.scatter(
                metrics_dict['feh_pred'][mask],
                metrics_dict['feh_true'][mask],
                c=colors[i],
                label=class_name,
                alpha=0.6,
                s=10
            )
        
        min_val = min(np.min(metrics_dict['feh_pred']), np.min(metrics_dict['feh_true']))
        max_val = max(np.max(metrics_dict['feh_pred']), np.max(metrics_dict['feh_true']))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
        
        for boundary in [-3, -2, -1]:
            ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title('FeH Prediction vs Truth')
        ax.set_xlabel('Predicted FeH')
        ax.set_ylabel('True FeH')
        ax.legend(loc='lower right')
        ax.grid(True)

        # Add Metrics Annotation
        accuracy = metrics_dict.get('accuracy', float('nan'))
        macro_f1 = metrics_dict.get('macro_f1', float('nan'))
        metrics_text = f"Accuracy: {accuracy:.4f}\nMacro F1: {macro_f1:.4f}"
        ax.text(0.95, 0.95, metrics_text,
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', fc='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(classification_dir, 'feh_prediction_vs_truth.pdf'))
        plt.close(fig)