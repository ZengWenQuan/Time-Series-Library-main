import torch
import torch.nn as nn

def select_criterion(loss_name: str):
    """
    根据名称选择损失函数。

    Args:
        loss_name (str): 损失函数的名称（不区分大小写）。
                         支持的值: 'Mse', 'l2', 'Mae', 'l1', 
                         'SmoothL1', 'Huber', 'LogCosh'。

    Returns:
        一个PyTorch损失函数实例或一个自定义的损失函数。
        对于未知的名称，默认为MSELoss。
    """
    loss_name = loss_name.lower()
    if loss_name == 'mse' or loss_name == 'l2':
        return nn.MSELoss()
    elif loss_name == 'mae' or loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'smoothl1':
        return nn.SmoothL1Loss()
    elif loss_name == 'huber':
        return nn.HuberLoss(delta=1.0)
    elif loss_name == 'logcosh':
        def logcosh_loss(pred, target):
            return torch.mean(torch.log(torch.cosh(pred - target)))
        return logcosh_loss
    else:
        print(f"警告: 未知的损失函数 '{loss_name}'，使用默认的MSE损失")
        return nn.MSELoss()
