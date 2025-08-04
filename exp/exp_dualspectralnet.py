from exp.exp_spectral_prediction import Exp_Spectral_Prediction

class Exp_DualSpectralNet(Exp_Spectral_Prediction):
    """
    DualSpectralNet的实验类，继承自Exp_Spectral_Prediction
    专门用于DualSpectralNet模型的训练和测试
    
    DualSpectralNet特点：
    - 双分支架构：连续谱分支 + 吸收线分支
    - 连续谱分支：CNN+Transformer提取全局特征
    - 吸收线分支：多尺度CNN提取局部特征  
    - 交叉注意力特征融合
    - 混合精度训练支持（继承自基础类）
    """
    def __init__(self, args):
        super(Exp_DualSpectralNet, self).__init__(args)
        
        # 验证模型配置
        if not hasattr(args, 'model_conf') or not args.model_conf:
            raise ValueError("DualSpectralNet requires a model configuration file specified via --model_conf")
        
        print(f"DualSpectralNet实验初始化完成")
        print(f"模型配置文件: {args.model_conf}")
        print(f"特征维度: {args.feature_size}")
        print(f"标签维度: {args.label_size}")
        
    def _build_model(self):
        """构建DualSpectralNet模型"""
        model = super(Exp_DualSpectralNet, self)._build_model()
        
        # 打印模型参数量（模型内部会自动打印详细信息）
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"DualSpectralNet模型构建完成:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        return model