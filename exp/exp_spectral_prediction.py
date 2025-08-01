
from exp.exp_basic import Exp_Basic
from models import MLP
from data_provider.data_factory import data_provider
import torch
import torch.nn as nn

class Exp_Spectral_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Exp_Spectral_Prediction, self).__init__(args)

    def _build_model(self):
        model = MLP.Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        return super().train()

    def test(self, setting, test=0):
        return super().test(setting, test=test)

    def predict(self, setting, load=False):
        return super().predict(setting, load=load)
