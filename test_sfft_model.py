import torch
import numpy as np
from models.regression.SFFTDualBranchNet import Model

# 创建一个简单的配置类
class Config:
    def __init__(self):
        self.feature_size = 1024  # 输入序列长度
        self.label_size = 3       # 输出标签数量

# 测试模型
def test_sfft_model():
    print("开始测试SFFTDualBranchNet模型...")
    
    # 创建配置
    config = Config()
    
    # 创建模型
    try:
        model = Model(config)
        print(f"✓ 模型创建成功")
        print(f"  - 输入特征大小: {config.feature_size}")
        print(f"  - 输出标签大小: {config.label_size}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数数量: {total_params:,}")
    print(f"  - 可训练参数数量: {trainable_params:,}")
    
    # 创建测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, config.feature_size)
    
    print(f"\n测试前向传播...")
    print(f"  - 输入形状: {test_input.shape}")
    
    # 前向传播测试
    try:
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ 前向传播成功")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return
    
    # 测试梯度计算
    print(f"\n测试反向传播...")
    try:
        model.train()
        output = model(test_input)
        loss = torch.mean(output ** 2)  # 简单的损失函数
        loss.backward()
        print(f"✓ 反向传播成功")
        print(f"  - 损失值: {loss.item():.6f}")
        
        # 检查梯度
        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"  - 梯度范数: {grad_norm:.6f}")
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        return
    
    print(f"\n🎉 所有测试通过！模型可以正常使用。")
    
    # 打印模型结构概览
    print(f"\n模型结构概览:")
    print(f"  1. SFFT特征提取器")
    print(f"  2. 全卷积分支 (5层卷积+池化)")
    print(f"  3. Inception分支 (5层多核卷积+池化)")
    print(f"  4. 特征融合与FFN输出")

if __name__ == "__main__":
    test_sfft_model()