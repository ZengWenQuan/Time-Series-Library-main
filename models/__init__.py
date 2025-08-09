
# Import all models from the 'spectral_prediction' subdirectory
from .spectral_prediction.MLP import MLP
from .spectral_prediction.MPBDNet_spetral import MPBDNet_spetral
from .spectral_prediction.SpectralMPBDNet import SpectralMPBDNet
from .spectral_prediction.TwoBranchTeffNet import TwoBranchTeffNet
from .spectral_prediction.DualPyramidNet import DualPyramidNet
from .spectral_prediction.MultiScaleSpectralTransformer import MultiScaleSpectralTransformer
from .spectral_prediction.CustomFusionNet import CustomFusionNet
from .spectral_prediction import FreqInceptionLNet, FreqInceptionConvNet, FreqInceptionNet, MLP, SpectralMPBDNet, TwoBranchTeffNet, DualPyramidNet, DualSpectralNet, DualBranchMoENet
