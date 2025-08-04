from exp.exp_spectral_prediction import Exp_Spectral_Prediction

class Exp_DualPyramidNet(Exp_Spectral_Prediction):
    """
    DualPyramidNet的实验类，继承自Exp_Spectral_Prediction
    专门用于DualPyramidNet模型的训练和测试
    """
    def __init__(self, args):
        super(Exp_DualPyramidNet, self).__init__(args)
        
    # 可以在这里添加DualPyramidNet特有的实验逻辑
    # 目前直接使用父类的所有功能