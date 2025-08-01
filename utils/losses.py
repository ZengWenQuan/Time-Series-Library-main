# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class RegressionFocalLoss(nn.Module):
    """
    回归任务的Focal Loss
    
    参数:
    - alpha: 平衡因子，控制正负样本的权重
    - gamma: 聚焦参数，控制难易样本的权重
    - reduction: 损失聚合方式，'mean'、'sum'或'none'
    - threshold: 误差阈值，超过此阈值的样本被视为难样本
    - base_loss: 基础损失函数，可以是'mse'、'mae'或'smooth_l1'
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', threshold=0.5, base_loss='mse'):
        super(RegressionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold
        
        # 选择基础损失函数
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss(reduction='none')
        elif base_loss == 'smooth_l1':
            self.base_criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"不支持的基础损失函数: {base_loss}")
    
    def forward(self, pred, target):
        """
        计算回归Focal Loss
        
        参数:
        - pred: 预测值 [batch_size, num_targets]
        - target: 目标值 [batch_size, num_targets]
        
        返回:
        - loss: 损失值
        """
        # 计算基础损失
        base_loss = self.base_criterion(pred, target)
        
        # 计算误差权重
        # 误差越大，权重越小（对于gamma > 0）
        error_ratio = t.clamp(base_loss / self.threshold, min=0, max=1)
        weights = self.alpha * (1 - error_ratio) ** self.gamma
        
        # 应用权重到基础损失
        focal_loss = weights * base_loss
        
        # 根据reduction方式聚合损失
        if self.reduction == 'mean':
            return t.mean(focal_loss)
        elif self.reduction == 'sum':
            return t.sum(focal_loss)
        else:  # 'none'
            return focal_loss
