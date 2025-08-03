import os
import torch
from models.other import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer
from models.regression import PatchSpectralModel, Inception1DModel, InceptionStellar, WaveletConvNet, MPBDNet, SFFTDualBranchNet, hybrid_model, AttentionSpectrumNet
from models.spectral_prediction import MLP ,MPBDNet_spetral
import torch.nn as nn
from utils.losses import RegressionFocalLoss

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'PatchSpectralModel': PatchSpectralModel,
            'Inception1DModel': Inception1DModel,
            'InceptionStellar': InceptionStellar,
            'WaveletConvNet': WaveletConvNet,
            'MPBDNet': MPBDNet,
            'SFFTDualBranchNet': SFFTDualBranchNet,
            'HybridModel': hybrid_model,
            'AttentionSpectrumNet': AttentionSpectrumNet,
            'MLP':MLP,
            'MPBDNet_spetral':MPBDNet_spetral
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _select_criterion(self):
        if not hasattr(self.args, 'loss') or self.args.loss == 'MSE':
            return nn.MSELoss()
        elif self.args.loss == 'MAE':
            return nn.L1Loss()
        elif self.args.loss == 'SmoothL1':
            return nn.SmoothL1Loss()
        elif self.args.loss == 'Huber':
            return nn.HuberLoss(delta=1.0)
        elif self.args.loss == 'LogCosh':
            def logcosh_loss(pred, target):
                return torch.mean(torch.log(torch.cosh(pred - target)))
            return logcosh_loss
        elif self.args.loss == 'FocalMSE':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mse')
        elif self.args.loss == 'FocalMAE':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='mae')
        elif self.args.loss == 'FocalSmoothL1':
            alpha = self.args.focal_alpha if hasattr(self.args, 'focal_alpha') else 1.0
            gamma = self.args.focal_gamma if hasattr(self.args, 'focal_gamma') else 2.0
            threshold = self.args.focal_threshold if hasattr(self.args, 'focal_threshold') else 0.5
            return RegressionFocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, base_loss='smooth_l1')
        else:
            print(f"警告: 未知的损失函数 '{self.args.loss}'，使用默认的MSE损失")
            return nn.MSELoss()