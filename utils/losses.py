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
import torch

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


class GaussianNLLLoss(nn.Module):
    """高斯负对数似然损失函数"""
    def __init__(self, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, true):
        # pred: [B, L, 2] (mean, log_var)
        # true: [B, L]
        mean = pred[..., 0]
        log_var = pred[..., 1]
        
        # 确保方差为正
        var = torch.exp(log_var)
        
        # 计算损失
        log_likelihood = -0.5 * (log_var + (true - mean)**2 / var)
        loss = -log_likelihood
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class RegressionFocalLoss(nn.Module):
    """
    Focal Loss for Regression.

    This is a PyTorch implementation of the Focal Loss for regression tasks,
    designed to focus training on "hard" samples by assigning them more weight.
    The loss is calculated as:
        loss = |y_true - y_pred|^gamma * L1Loss(y_true, y_pred)

    Args:
        gamma (float): The focusing parameter. Higher values of gamma give more
                       weight to hard-to-predict samples. Default: 1.5
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, gamma=1.5, reduction='mean'):
        super(RegressionFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # We use L1Loss with no reduction to get the per-sample error
        self.base_criterion = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        """
        Forward pass.

        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The calculated focal regression loss.
        """
        # 1. Calculate the absolute error for each sample
        l1_loss = self.base_criterion(pred, target)

        # 2. Create the modulating factor (focal weight)
        # This gives higher weight to samples with larger errors.
        focal_weight = torch.pow(l1_loss, self.gamma)

        # 3. The final loss is the element-wise product of the focal weight and the L1 loss
        focal_loss = focal_weight * l1_loss

        # 4. Apply the specified reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss
